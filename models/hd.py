import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_add
from utils.atom_utils import remove_mean, backmapping_matrix, mapping_matrix, kabsch_torch_batched


class EquivariantHeatDissipation(nn.Module):
    def __init__(self, config, args, encoder):
        super().__init__()
        self.T = args.num_steps
        self.sigma = args.sigma
        self.encoder = encoder

    def fwd_blur(self, batch, t_steps, blur_t, x_a, x_f_ref, bm_mat):
        b_list, lb_list = list(), list()
        
        x_a_gt = remove_mean(x_a, batch.batch)
        x_f_ref_ext = bm_mat@x_f_ref

        n_nodes_per_g = batch.num_nodes_per_graph
        x_a_gt_split = list(torch.split(x_a_gt, n_nodes_per_g.tolist(), dim=0))
        x_f_ref_ext_split = list(torch.split(x_f_ref_ext, n_nodes_per_g.tolist(), dim=0))

        b_t, lb_t = blur_t[t_steps], blur_t[t_steps-1]
        for i, _ in enumerate(batch.fg2cg):
            b_list.append(torch.lerp(x_a_gt_split[i], x_f_ref_ext_split[i], b_t[i].to(x_a.device)))
            lb_list.append(torch.lerp(x_a_gt_split[i], x_f_ref_ext_split[i], lb_t[i].to(x_a.device)))
            
        b_batch = torch.cat([b_list[i] for i in range(len(n_nodes_per_g))], dim=0)
        lb_batch = torch.cat([lb_list[i] for i in range(len(n_nodes_per_g))], dim=0)
    
        return b_batch, lb_batch
    
    def sigma_blur(self, num_t_steps):
        diss_time = [i/num_t_steps for i in range(num_t_steps)]
        return nn.Parameter(torch.tensor(diss_time), requires_grad=False)
    
    def t_sample(self, batch_size, device, K):
        return torch.randint(1, K, (batch_size,), device=device)

    def get_loss(self, batch):
        # prepare coordinates.
        m_mat = mapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
        bm_mat = backmapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
        x_a_ref = remove_mean(batch.ref_pos, batch.batch)
        x_a_gt = remove_mean(batch.alg_gt_pos, batch.batch)
        x_f_gt, x_f_ref = m_mat@x_a_gt, m_mat@x_a_ref
        x_f_gt_ext = bm_mat@x_f_gt
        x_r_gt = x_a_gt - x_f_gt_ext

        # perturb data using scheduler.
        blur_t = self.sigma_blur(self.T)
        t_steps = self.t_sample(batch.batch[-1].item()+1, batch.batch.device, self.T)
        x_a_b, x_a_lb = self.fwd_blur(batch, t_steps, blur_t, x_a_gt, x_f_ref, bm_mat)
        n_ = torch.randn_like(x_a_b)
        n = remove_mean(n_, batch.batch)
        x_a_b, x_a_lb = remove_mean(x_a_b, batch.batch), remove_mean(x_a_lb, batch.batch)
        x_a_b = x_a_b + n * self.sigma

        # prediction.
        t = t_steps / self.T
        pred_a, _ = self.encoder._forward(t, batch, x_a_b, m_mat@x_a_b, m_mat, bm_mat)        

        # align prediction to less blurred atom position.  
        n_nodes_per_g = batch.num_nodes_per_graph
        pred__ = list(torch.split(pred_a, n_nodes_per_g.tolist(), 0))
        pred_batch = pad_sequence(pred__, batch_first=True, padding_value=0.)
        x_a_gt__ = list(torch.split(x_a_gt, n_nodes_per_g.tolist(), 0))
        x_a_gt_batch = pad_sequence(x_a_gt__, batch_first=True, padding_value=0.)
        alg_pred_batch_to_gt = kabsch_torch_batched(pred_batch, x_a_gt_batch)
        alg_pred_to_gt = torch.cat([alg_pred_batch_to_gt[i][:n_nodes_per_g[i]] for i in range(len(n_nodes_per_g))], dim=0)

        # loss computation.
        loss_a_alg_ = (alg_pred_to_gt - x_a_gt) ** 2
        loss_a_alg_ = torch.sum(scatter_add(loss_a_alg_.T, batch.batch).T, dim=-1)
        loss_a_alg = torch.mean(loss_a_alg_)

        return loss_a_alg, t_steps
    
    def sample(self, sample_init, batch, delta):
        with torch.no_grad():
            K = self.T
            m_mat = mapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
            bm_mat = backmapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
            
            sample_gen_traj = []            
            sample_init = remove_mean(sample_init, batch.batch)

            sample_init__ = list(torch.split(sample_init, batch.num_nodes_per_graph.tolist(), 0))
            sample_init_batch = pad_sequence(sample_init__, batch_first=True, padding_value=0.)
            
            x_f = m_mat@sample_init
            u = bm_mat@x_f
            u = remove_mean(u, batch.batch)
            noise = torch.randn_like(u)
            noise = remove_mean(noise, batch.batch)
            u = u + noise*delta

            if sample_gen_traj != None and K in sample_gen_traj:
                sample_gen_traj.append((u, u))
            
            blur_t = self.sigma_blur(K)
            for i in range(K-1, 0, -1):
                t = torch.ones(batch.batch[-1].item()+1, device=u.device, dtype=torch.long) * blur_t[i]
                pred, _ = self.encoder._forward(t, batch, u, m_mat@u, m_mat, bm_mat) 
                noise = torch.randn_like(u)
                noise = remove_mean(noise, batch.batch)
                t_steps = torch.ones(batch.batch[-1].item()+1, device=u.device, dtype=torch.long) * i

                pred__ = list(torch.split(pred, batch.num_nodes_per_graph.tolist(), 0))
                pred_batch = pad_sequence(pred__, batch_first=True, padding_value=0.)
                alg_pred_batch_to_init = kabsch_torch_batched(pred_batch, sample_init_batch)
                alg_pred_to_init = torch.cat([alg_pred_batch_to_init[i][:batch.num_nodes_per_graph[i]] for i in range(len(batch.num_nodes_per_graph))], dim=0)
                _, u_ = self.fwd_blur(batch, t_steps, blur_t, alg_pred_to_init, x_f, bm_mat)

                u = u_ + noise*delta

                u = remove_mean(u, batch.batch)
                sample_gen_traj.append((u, pred))

        return u, [u for (u, pred) in sample_gen_traj]

