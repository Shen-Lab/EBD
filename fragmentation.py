import os
import argparse
import yaml
from easydict import EasyDict
from glob import glob
import torch

from utils.frag_utils import *
from utils.data_utils import *
from utils.misc import *

import pickle

import rdkit.Chem as Chem
from torch_geometric.utils import contains_isolated_nodes


def cluster_matrix(assignment, fg_size):
    assign_mat = torch.zeros(size=[fg_size, len(assignment)])
    for i in range(len(assignment)):
        assign_mat[assignment[i], i] = 1
    assign_mat = torch.t(assign_mat)
    assign_mat = assign_mat / (torch.sum(assign_mat, 1).unsqueeze(-1))
    assign_mat = assign_mat.to_sparse()
    return assign_mat


def vocab_gen_ps(smis, rdmols, vocab_len, vocab_path, cpus, kekulize):
    # loop
    mols = [MolInSubgraph(mol, kekulize) for mol in rdmols]
    selected_smis, details = list(MAX_VALENCE.keys()), {}   # details: <smi: [atom cnt, frequency]
    # calculate single atom frequency
    for atom in selected_smis:
        details[atom] = [1, 0]  # frequency of single atom is not calculated
    for smi in smis:
        cnts = cnt_atom(smi, return_dict=True)
        for atom in details:
            if atom in cnts:
                details[atom][1] += cnts[atom]
    # bpe process
    add_len = vocab_len - len(selected_smis)
    # pbar = tqdm(total=add_len)
    pool = mp.Pool(cpus)
    while len(selected_smis) < vocab_len:
        res_list = pool.map(freq_cnt, mols)  # each element is (freq, mol) (because mol will not be synced...)
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # find the subgraph to merge
        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            cnt = freqs[smi]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_smi = smi
        # merge
        for mol in mols:
            mol.merge(merge_smi)
        if merge_smi in details:  # corner case: re-extracted from another path
            continue
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
        # pbar.update(1)
    # pbar.close()
    print_log('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    with open(vocab_path, 'w') as fout:
        fout.write(json.dumps({'kekulize': kekulize}) + '\n')
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    return selected_smis, details


def fragment_ps(tokenizer, mols, vocab_len, indices):
    train_end_idx, val_end_idx = indices[0], indices[1]
    for idx, mol in enumerate(mols):
        n_atoms = mol.GetNumAtoms()
        if idx < train_end_idx:
            graph = train_set.data[idx]
        elif train_end_idx <= idx < val_end_idx:
            graph = val_set.data[idx-train_end_idx]
        elif val_end_idx <= idx:
            graph = test_set.data[idx-val_end_idx]

        frag_list = tokenizer.tokenize(mol)

        cg, cg_smiles = [], []
        for frag in frag_list:
            cg_smiles.append(frag[0])
            cg.append(frag[1])

        fg2cg = cluster_matrix(cg, n_atoms)
        cg_pos = torch.matmul(fg2cg, graph.pos)
        graph.fg2cg = cg
        graph.cg_pos = cg_pos
        graph.cg_smile = cg_smiles
        graph.num_frags = len(cg)
    
    train_data_path = os.path.join(os.path.dirname(vocab_path), 'train_data_40k_ps'+str(vocab_len)+'.pkl')
    val_data_path = os.path.join(os.path.dirname(vocab_path), 'val_data_5k_ps'+str(vocab_len)+'.pkl')
    test_data_path = os.path.join(os.path.dirname(vocab_path), 'test_data_1k_ps'+str(vocab_len)+'.pkl')
    with open(train_data_path, "wb") as fout:
        pickle.dump(train_set.data, fout)
    with open(val_data_path, 'wb') as fout:
        pickle.dump(val_set.data, fout)
    with open(test_data_path, 'wb') as fout:
        pickle.dump(test_set.data, fout)
    
    return train_set, val_set, test_set
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vocab_len', type=int, default=500, help='size of vocab')
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
transforms = CountNodesPerGraph()
test_transforms = Compose([CountNodesPerGraph()])
train_set = ConformationDataset(config.dataset.train, transform=transforms)
val_set = ConformationDataset(config.dataset.val, transform=transforms)
test_set = PackedConformationDataset(config.dataset.test, transform=test_transforms)
train_end_idx, val_end_idx = train_set.__len__(), train_set.__len__()+val_set.__len__()
data_indices = [train_end_idx, val_end_idx]

smis = [Chem.MolToSmiles(train_set.data[i].rdmol) for i in range(len(train_set))] + \
         [Chem.MolToSmiles(val_set.data[i].rdmol) for i in range(len(val_set))] + \
            [Chem.MolToSmiles(test_set.data[i].rdmol) for i in range(len(test_set.data))]
mols = [train_set.data[i].rdmol for i in range(len(train_set))] + \
            [val_set.data[i].rdmol for i in range(len(val_set))] + \
                [test_set.data[i].rdmol for i in range(len(test_set.data))]


# Vocabulary generation
vocab_path = os.path.join(os.path.dirname(config.dataset.train), 'vocab_'+args.frag+str(args.vocab_len)+'.pkl')
vocab_gen_ps(smis=smis, rdmols=mols, vocab_path=vocab_path, vocab_len=args.vocab_len, cpus=args.cpus, kekulize=args.kekulize)
tokenizer = Tokenizer(vocab_path)
fragment_ps(tokenizer=tokenizer, mols=mols, vocab_len=args.vocab_len, indices=data_indices)