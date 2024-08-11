import torch
import torch.nn as nn
from torch_sparse import coalesce
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph, radius
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils.chem_utils import BOND_TYPES
import sys


### https://hunterheidenreich.com/posts/kabsch_algorithm/
def kabsch_torch_batched(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    ## From EquiDock
    A = P.transpose(1, 2) @ Q

    assert not torch.isnan(A).any()
    U, S, Vt = torch.linalg.svd(A)

    # corr_mat = torch.diag(torch.Tensor([1,1,torch.sign(torch.det(A))])).to(P.device)
    corr_mat_ = torch.diag(torch.Tensor([1, 1, 0])).to(P.device)
    corr_mat = corr_mat_.repeat(U.size(0), 1, 1)
    corr_mat[:,2,2] = torch.sign(torch.det(A))
    R = (U @ corr_mat) @ Vt

    P_align = P @ R

    return P_align


def assignment_matrix(assignment, fg_size):
    assign_mat = torch.zeros(size=[fg_size, len(assignment)])
    for i in range(len(assignment)):
        assign_mat[assignment[i], i] = 1
    return assign_mat


def backmapping_matrix(fg2cg, num_atoms):
    backmap_mat = assignment_matrix(fg2cg[0], num_atoms[0])
    for i in range(len(fg2cg)-1):
        backmap_mat_ = assignment_matrix(fg2cg[i+1], num_atoms[i+1])
        backmap_mat = torch.block_diag(backmap_mat, backmap_mat_)
    backmap_mat = backmap_mat.to_sparse().to(num_atoms.device)
    return backmap_mat


def cluster_matrix(assignment, fg_size):
    cluster_mat = torch.zeros(size=[fg_size, len(assignment)])
    for i in range(len(assignment)):
        cluster_mat[assignment[i], i] = 1
    cluster_mat = torch.t(cluster_mat)
    cluster_mat = cluster_mat / (torch.sum(cluster_mat, 1).unsqueeze(-1))
    return cluster_mat


def mapping_matrix(fg2cg, num_atoms):
    map_mat = cluster_matrix(fg2cg[0], num_atoms[0])
    for i in range(len(fg2cg)-1):
        map_mat_ = cluster_matrix(fg2cg[i+1], num_atoms[i+1])
        map_mat = torch.block_diag(map_mat, map_mat_)
    map_mat = map_mat.to_sparse().to(num_atoms.device)
    return map_mat


def remove_mean(x, batch):
    mean = scatter_mean(x.T, batch).T
    x = x - mean[batch]
    return x


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)   # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
    
    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added. 
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type
    

def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=0, is_sidechain=None):

    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(
        edge_index, 
        edge_type, 
        torch.Size([N, N])
    )

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)    # (2, E_r)
    else:
        # fetch sidechain and its batch index
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch)
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y)) # (2, E)
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x)) # (2, E)
        rgraph_edge_index = torch.cat((rgraph_edge_index1, rgraph_edge_index2), dim=-1) # (2, 2E)
        # delete self loop
        rgraph_edge_index = rgraph_edge_index[:, (rgraph_edge_index[0] != rgraph_edge_index[1])]

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index, 
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    
    return new_edge_index, new_edge_type


def extend_graph_order_radius(num_nodes, pos, edge_index, edge_type, batch, order=3, cutoff=10.0, 
                              extend_order=True, extend_radius=True, is_sidechain=None):
    
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes, 
            edge_index=edge_index, 
            edge_type=edge_type, order=order
        )
        # edge_index_order = edge_index
        # edge_type_order = edge_type

    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos, 
            edge_index=edge_index, 
            edge_type=edge_type, 
            cutoff=cutoff, 
            batch=batch,
            is_sidechain=is_sidechain

        )
    
    return edge_index, edge_type


def cumsum(x, dim=0):
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))
    return out