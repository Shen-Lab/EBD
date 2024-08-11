from torch import nn
import torch
import math
from utils.atom_utils import backmapping_matrix


# Invarinat layer for feature update
class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.bond_emb = nn.Embedding(100, embedding_dim=hidden_nf) 
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf*2 + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_type=None):
        if edge_type is None:  
            out = torch.cat([source, target, edge_attr], dim=1)
        else:
            type_emb = self.bond_emb(edge_type)
            out = torch.cat([source, target, edge_attr, type_emb], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        return out, mij

    def node_model(self, target, aux, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=target.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([target, aux, agg, node_attr], dim=1)
        else:
            agg = torch.cat([target, aux, agg], dim=1)
        out = target + self.node_mlp(agg)
        return out, agg
    
    def forward(self, h_f, h_a, edge_index, edge_attr=None, node_attr=None, edge_type=None):
        row, col = edge_index
        if edge_type is not None:
            edge_feat, mij = self.edge_model(h_a[row], h_a[col], edge_attr, edge_type)
            h, _ = self.node_model(h_a, h_f, edge_index, edge_feat, node_attr)
        else:
            edge_feat, mij = self.edge_model(h_f[row], h_f[col], edge_attr, edge_type)
            h, _ = self.node_model(h_f, h_a, edge_index, edge_feat, node_attr)
        return h, mij


# Equivariant layer for coor update
class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        input_edge_type = hidden_nf * 2 + edges_in_d + hidden_nf

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.bond_emb = nn.Embedding(100, embedding_dim=hidden_nf) 
        self.coord_mlp_atom = nn.Sequential(
            nn.Linear(input_edge_type, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.coord_mlp_frag = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h_a, coord, e_a_idx, e_a_type, e_a_attr, coord_diff_a, h_f, x_f, bm_mat):
        e_a_row, e_a_col = e_a_idx
        bond_emb = self.bond_emb(e_a_type)

        coord_diff_f = coord - bm_mat@x_f
        radial_f = torch.sum((coord_diff_f) ** 2, 1).unsqueeze(1)
        norm_f = torch.sqrt(radial_f + 1e-8)
        coord_diff_f = coord_diff_f/(norm_f + 1)    

        input_tensor_atom = torch.cat([h_a[e_a_row], h_a[e_a_col], e_a_attr, bond_emb], dim=1)
        input_tensor_frag = torch.cat([h_a, bm_mat@h_f, torch.cat([radial_f, radial_f], dim=1)], dim=1)

        if self.tanh:
            trans_atom = coord_diff_a * torch.tanh(self.coord_mlp_atom(input_tensor_atom)) * self.coords_range
            trans_frag = coord_diff_f * torch.tanh(self.coord_mlp_frag(input_tensor_frag)) * self.coords_range
        else:
            trans_atom = coord_diff_a * self.coord_mlp_atom(input_tensor_atom)
            trans_frag = coord_diff_f * self.coord_mlp_frag(input_tensor_frag)

        agg_atom = unsorted_segment_sum(trans_atom, e_a_row, num_segments=coord.size(0),
                                        normalization_factor=self.normalization_factor,
                                        aggregation_method=self.aggregation_method)
        agg_frag = trans_frag

        coord = coord + agg_atom + agg_frag

        return coord

    def forward(self, h_a, x_a, e_a_idx, e_a_type, e_a_attr, coord_diff_a, h_f, x_f, bm_mat):
        coord = x_a
        coord = self.coord_model(h_a, coord, e_a_idx, e_a_type, e_a_attr, coord_diff_a, h_f, x_f, bm_mat)
        return coord


# Build block as a combination of GCL and EquivariantUpdate
class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, 
                 attention=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("f_inv_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                                act_fn=act_fn, attention=attention,
                                                normalization_factor=self.normalization_factor,
                                                aggregation_method=self.aggregation_method))
            self.add_module("a_inv_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf+self.hidden_nf,
                                                act_fn=act_fn, attention=attention,
                                                normalization_factor=self.normalization_factor,
                                                aggregation_method=self.aggregation_method))
        self.add_module("a_eq", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), 
                                                  tanh=tanh, coords_range=self.coords_range_layer,
                                                  normalization_factor=self.normalization_factor,
                                                  aggregation_method=self.aggregation_method))
        self.to(self.device)
    
    def atom_feat_pooling(self, m_mat, atom_feat):
        m_mat = m_mat.to_dense()
        m_mat[m_mat != 0] = 1
        h_a_agg = m_mat@atom_feat
        return h_a_agg

    def forward(self, h_a, x_a, e_a_idx, e_a_type, e_a_attr, h_f, x_f, e_f_idx, e_f_attr, m_mat, bm_mat):
        dist_f, _ = coord2diff(x_f, e_f_idx, self.norm_constant)
        dist_a, coord_diff_a = coord2diff(x_a, e_a_idx, self.norm_constant)
        if self.sin_embedding is not None:
            dist_f, dist_a = self.sin_embedding(dist_f), self.sin_embedding(dist_a)

        e_f_attr = torch.cat([dist_f, e_f_attr], dim=1)
        e_a_attr = torch.cat([dist_a, e_a_attr], dim=1)

        for i in range(0, self.n_layers):
            h_a_agg = self.atom_feat_pooling(m_mat, h_a)
            h_f, _ = self._modules["f_inv_%d" % i](h_f, h_a_agg, e_f_idx, edge_attr=e_f_attr)
            h_f_ext = bm_mat@h_f
            h_a, _ = self._modules["a_inv_%d" % i](h_f_ext, h_a, e_a_idx, edge_attr=e_a_attr, edge_type=e_a_type)
        coord = self._modules["a_eq"](h_a, x_a, e_a_idx, e_a_type, e_a_attr, coord_diff_a, h_f, x_f, bm_mat)
        return h_a, h_f, coord


# Build EGNN as a combination of EquivariantBlock
class EGNN(nn.Module):
    def __init__(self, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False, tanh=False, coords_range=15, 
                 norm_constant=1, inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbedding()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.atom_embedding = nn.Sequential(
            nn.Embedding(100, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf+1, self.hidden_nf),
            nn.SiLU())
        self.frag_embedding = nn.Sequential(
            nn.Linear(4, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.SiLU())
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device, act_fn=act_fn, 
                                                               n_layers=inv_sublayers, attention=attention, tanh=tanh, coords_range=coords_range, 
                                                               norm_constant=norm_constant, sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, t, a_type, x_a, e_a_idx, e_a_type, x_f, h_f, e_f_idx, m_mat, bm_mat, num_atoms, num_frags):
        h_a = self.atom_embedding[0:2](a_type)
        h_a_t = t.repeat_interleave(num_atoms)
        h_a = torch.cat([h_a, h_a_t.view(-1, 1)], dim=1)
        h_a = self.atom_embedding[2:](h_a)

        h_f_t = t.repeat_interleave(num_frags)
        h_f = torch.cat([h_f, h_f_t.view(-1, 1)], dim=1)
        h_f = self.frag_embedding(h_f)
        
        dist_f, _ = coord2diff(x_f, e_f_idx)
        dist_a, _ = coord2diff(x_a, e_a_idx)
        if self.sin_embedding is not None:
            dist_f, dist_a = self.sin_embedding(dist_f), self.sin_embedding(dist_a)

        for i in range(0, self.n_layers):
            h_a, h_f, x_a = self._modules["e_block_%d" % i](h_a, x_a, e_a_idx, e_a_type, dist_a, 
                                                            h_f, x_f, e_f_idx, dist_f, m_mat, bm_mat)
        return x_a 


class SinusoidsEmbedding(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
