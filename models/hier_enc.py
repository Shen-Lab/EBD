import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.utils import from_networkx
from .egnn import EGNN
from utils.atom_utils import extend_graph_order_radius, remove_mean


class HierarchicalMessagePassing(nn.Module):
    def __init__(self, args):
        super().__init__()    
        self.edge_order=args.edge_order
        self.cutoff=args.cutoff
        self.n_dims = 3
        self.hidden_nf = args.nf
        self.device = args.device
        self.act_fn = torch.nn.SiLU()
        self.n_layers = args.n_layers
        self.attention = args.attention
        self.tanh = args.tanh
        self.norm_constant = args.norm_constant
        self.inv_sublayers = args.inv_sublayers
        self.sin_embedding = args.sin_embedding
        self.normalization_factor = args.normalization_factor
        self.aggregation_method = args.aggregation_method
        self.extend_radius = args.extend_radius

        self.enc = EGNN(
            hidden_nf=self.hidden_nf, device=self.device, act_fn=self.act_fn, n_layers=self.n_layers, 
            attention=self.attention, tanh=self.tanh, norm_constant=self.norm_constant, inv_sublayers=self.inv_sublayers, 
            sin_embedding=self.sin_embedding, normalization_factor=self.normalization_factor, aggregation_method=self.aggregation_method)

    def forward(self, t, batch, x_a, x_f, m_mat, bm_mat):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def get_fully_connected(self, num_nodes):
        return from_networkx(G=nx.complete_graph(num_nodes)).edge_index.to(self.device)

    def get_frag_complete_edges(self, num_frags_per_graph):
        edges_frag = self.get_fully_connected(num_frags_per_graph[0])
        cum_num = num_frags_per_graph[0].item()
        for _, val in enumerate(num_frags_per_graph[1:]):
            edges_frag = torch.cat((edges_frag, torch.add(self.get_fully_connected(val), cum_num)), dim=1)
            cum_num += val
        return edges_frag

    def _forward(self, t, batch, x_a, x_f, m_mat, bm_mat):
        # frag feat
        Hydrophobicity = ['C']
        Hydrogen_bond_center = ['N', 'O', 'S', 'P']
        Negative_charge_cnter = ['F', 'Cl', 'Br', 'I']
        h_f = torch.zeros_like(x_f)
        ptr = 0
        for i in range(batch.num_graphs):
            graph = batch[i]
            a_symbol = []
            for atom in graph.rdmol.GetAtoms():
                a_symbol.append(atom.GetSymbol())
            for j, frag_lst in enumerate(graph.fg2cg):
                for idx in frag_lst:
                    if a_symbol[idx] in Hydrophobicity:
                        h_f[ptr+j, 0] += 1
                    elif a_symbol[idx] in Hydrogen_bond_center:
                        h_f[ptr+j, 1] += 1
                    elif a_symbol[idx] in Negative_charge_cnter:
                        h_f[ptr+j, 2] += 1
            ptr += j+1    
        # frag edge 
        e_f_idx = self.get_frag_complete_edges(batch.num_frags)
        # atom bond expansion
        e_a_idx, e_a_type = extend_graph_order_radius(
            num_nodes=x_a.shape[0],
            pos=x_a,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type, 
            batch=batch.batch,
            order=self.edge_order,
            cutoff=self.cutoff,
            extend_order=True,
            extend_radius=self.extend_radius,
            is_sidechain=None,
            )
        # forward pass
        pred = self.enc(t, batch.atom_type, x_a, e_a_idx, e_a_type, x_f, h_f, e_f_idx, m_mat, bm_mat,
                        batch.num_nodes_per_graph, batch.num_frags)    

        if torch.any(torch.isnan(pred)):
            print('Warning: detected nan in prediction, resetting encoder output to zero.')
            pred = torch.zeros_like(pred)

        x_a_deblur = remove_mean(pred, batch.batch)
        return x_a_deblur, e_f_idx



