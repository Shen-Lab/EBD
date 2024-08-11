import os
import argparse
import yaml
import torch
import pickle
import math
from tqdm import tqdm

from easydict import EasyDict
from glob import glob

from utils.data_utils import *
from utils.chem_utils import *
from utils.atom_utils import cluster_matrix, assignment_matrix
from utils.misc import *

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

from scipy.optimize import linear_sum_assignment
from rmsd import kabsch_rmsd


def rigid_transform_Kabsch_3D_torch(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


def get_multiple_rdkit_coords(molecule, num_conf):
    mol = copy.deepcopy(molecule)
    mol.RemoveAllConformers()
    ps = AllChem.ETDG()
    ps.maxIterations = 5000
    ps.randomSeed = 2023
    ps.useBasicKnowledge = False
    ps.useExpTorsionAnglePrefs = False
    ps.useRandomCoords = False
    ids = AllChem.EmbedMultipleConfs(mol, num_conf, ps)
    if -1 in ids or mol.GetNumConformers() != num_conf:
        print("Use DG random coords.")
        ps.useRandomCoords = True
        ids = AllChem.EmbedMultipleConfs(mol, num_conf, ps)
    confs = []
    for cid in range(num_conf):
        confs.append(torch.tensor(mol.GetConformer(cid).GetPositions(), dtype=torch.float32))
    return confs


def revised_subgraph(subset, edge_index, edge_attr=None, num_nodes=None):
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    n_mask = torch.zeros(num_nodes, dtype=torch.bool)
    n_mask[subset] = 1

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
    n_idx[subset] = torch.arange(subset.size(0), device=device)

    return n_idx, edge_index, edge_attr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--cpus', type=int, default=16)
    parser.add_argument('--kekulize', type=bool, default=False)
    args = parser.parse_args()

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)

    # Loading dataset
    ps50_train_set_path = os.path.join(os.path.dirname(config.dataset.train), 'train_data_40k_ps50.pkl')
    ps50_val_set_path = os.path.join(os.path.dirname(config.dataset.train), 'val_data_5k_ps50.pkl')
    ps50_test_set_path = os.path.join(os.path.dirname(config.dataset.train), 'test_data_1k_ps50.pkl')

    transforms = CountNodesPerGraph()
    test_transforms = Compose([CountNodesPerGraph(),])
    ps50_train_set = ConformationDataset(ps50_train_set_path, transform=transforms)
    ps50_val_set = ConformationDataset(ps50_val_set_path, transform=transforms)
    ps50_test_set = PackedConformationDataset(ps50_test_set_path, transform=test_transforms)

    train_end_idx, val_end_idx = ps50_train_set.__len__(), ps50_train_set.__len__() + ps50_val_set.__len__()
    
    # train / val
    for idx in tqdm(range(0, val_end_idx, 5)):
        
        gt_confs = []
        if idx < train_end_idx:
            graph = ps50_train_set.data[idx]
            mol = Chem.Mol(graph.rdmol)
            ref_confs = get_multiple_rdkit_coords(mol, 5)
            for i in range(5):
                gt_confs.append(ps50_train_set.data[idx+i].pos)
        elif train_end_idx <= idx < val_end_idx:
            graph = ps50_val_set.data[idx-train_end_idx]
            mol = Chem.Mol(graph.rdmol)
            ref_confs = get_multiple_rdkit_coords(mol, 5)
            for i in range(5):
                gt_confs.append(ps50_val_set.data[idx-train_end_idx+i].pos)

        map_mat = cluster_matrix(graph.fg2cg, len(graph.atom_type))
        backmap_mat = assignment_matrix(graph.fg2cg, len(graph.atom_type))
        cost_matrix = [[kabsch_rmsd((backmap_mat@(map_mat@gt_confs[i])).numpy(), (backmap_mat@(map_mat@ref_confs[j])).numpy(), translate=True) 
                        for j in range(5)] for i in range(5)]
        cost_matrix = np.asarray(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for j in range(5):
            if idx < train_end_idx:
                ps50_graph = ps50_train_set.data[idx+j]
            elif train_end_idx <= idx < val_end_idx:
                ps50_graph = ps50_val_set.data[idx-train_end_idx+j]

            ps50_cg_intra_edge = []
            for i, node_subset in enumerate(ps50_graph.fg2cg):
                n_idx, edge_subset, _ = revised_subgraph(node_subset, ps50_graph.edge_index, num_nodes=len(ps50_graph.atom_type))
                ps50_cg_intra_edge.append(edge_subset)  

            ps50_graph.cg_intra_edge = ps50_cg_intra_edge

            ref_pos = ref_confs[col_ind[j]]
            R, t = rigid_transform_Kabsch_3D_torch(ps50_graph.pos.T, ref_pos.T)
            alg_gt_pos = R@(ps50_graph.pos.T)+t

            ps50_graph.ref_pos = ref_pos
            ps50_graph.alg_gt_pos = alg_gt_pos.T

    ps50_train_data_path = os.path.join(os.path.dirname(config.dataset.train), 'train_data_40k_ps50_dg.pkl')
    ps50_val_data_path = os.path.join(os.path.dirname(config.dataset.train), 'val_data_5k_ps50_dg.pkl')

    with open(ps50_train_data_path, "wb") as fout:
        pickle.dump(ps50_train_set.data, fout)

    with open(ps50_val_data_path, 'wb') as fout:
        pickle.dump(ps50_val_set.data, fout)
        
    # test
    for idx in tqdm(range(len(ps50_test_set.data))):
        ps50_graph = ps50_test_set.data[idx]

        ps50_cg_intra_edge = []
        for i, node_subset in enumerate(ps50_graph.fg2cg):
            n_idx, edge_subset, _ = revised_subgraph(node_subset, ps50_graph.edge_index, num_nodes=len(ps50_graph.atom_type))
            ps50_cg_intra_edge.append(edge_subset)  

        ps50_graph.cg_intra_edge = ps50_cg_intra_edge

    ps50_test_data_path = os.path.join(os.path.dirname(config.dataset.train), 'test_data_1k_ps50_dg.pkl')

    with open(ps50_test_data_path, 'wb') as fout:
        pickle.dump(ps50_test_set.data, fout)








