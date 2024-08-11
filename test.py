import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom

from models import get_model
from utils.data_utils import *
from utils.misc import *


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
        confs.append(mol.GetConformer(cid).GetPositions())
    return np.array(confs)


def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true', default=False, help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--delta', type=float, default=0.0125, help='stdev of noise')
    test_args, unparsed_args = parser.parse_known_args()

    # Load checkpoint
    ckpt = torch.load(test_args.ckpt)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(test_args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    arg_path = os.path.join(os.path.dirname(os.path.dirname(test_args.ckpt)), 'args.pickle')
    with open(arg_path, 'rb') as f:
        args = pickle.load(f)
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(test_args.ckpt))

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=test_args.tag)
    logger = get_logger('test', output_dir)
    logger.info(test_args)

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        # AddHigherOrderEdges(order=args.edge_order), # Offline edge augmentation
    ])
    test_set_path = os.path.join(os.path.dirname(config.dataset.test), 'test_data_1k_'+args.frag+str(args.vocab_len)+'_dg.pkl')

    if test_args.test_set is None:
        test_set = PackedConformationDataset(test_set_path, transform=transforms)
    else:
        test_set = PackedConformationDataset(test_args.test_set, transform=transforms)
    
    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (test_args.start_idx <= i < test_args.end_idx): continue
        test_set_selected.append(data)

    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'], args).to(args.device)
    if args.ema_decay > 0:
        model.load_state_dict(ckpt['model_ema'])
    else:
        model.load_state_dict(ckpt['model'])

    done_smiles = set()
    results = []
    if test_args.resume is not None:
        with open(test_args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)
    
    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_smiles:
            logger.info('Molecule#%d is already done.' % i)
            continue

        num_refs = data.pos_ref.size(0) // data.num_nodes
        num_samples = test_args.num_confs(num_refs)
        data_input = data.clone()
        data_input['pos_ref'] = None
        batch = repeat_data(data_input, num_samples).to(args.device)

        for _ in range(2):
            try:
                rdkit_confs = get_multiple_rdkit_coords(data_input.rdmol, num_samples)
                rdkit_confs = torch.as_tensor(rdkit_confs, dtype=torch.float32).to(args.device)
                sample_init = torch.reshape(rdkit_confs, (batch.num_nodes, 3))
                sample_gen, sample_gen_traj = model.sample(sample_init, batch, test_args.delta)

                sample_gen = sample_gen.cpu()
                if test_args.save_traj:
                    data.sample_gen = torch.stack(sample_gen_traj)
                else:
                    data.sample_gen = sample_gen
                results.append(data)
                done_smiles.add(data.smiles)

                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)
                break
            except FloatingPointError:
                logger.warning('Retrying')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
        
    