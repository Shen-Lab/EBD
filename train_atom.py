import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch_geometric.data import DataLoader
import torch_geometric as tg
import networkx as nx

from models import get_model
from utils.data_utils import *
from utils.misc import *
from utils.opt_utils import get_optimizer, EMA, Queue, gradient_clipping
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--type', type=str, default='hd')
    parser.add_argument('--num_steps', type=int, default=50, help='number of time steps')
    parser.add_argument('--sigma', type=float, default=0.01, help='noise scale')
    parser.add_argument('--vel', type=eval, default=False)
    parser.add_argument('--frag', type=str, default='ps')
    parser.add_argument('--vocab_len', type=int, default=50, help='size of vocab')
    parser.add_argument('--n_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--inv_sublayers', type=int, default=1, help='number of sublayers')
    parser.add_argument('--nf', type=int, default=128, help='feature dimension')
    parser.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    parser.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
    parser.add_argument('--norm_constant', type=float, default=1, help='diff/(|diff| + norm_constant)')
    parser.add_argument('--condition_time', type=eval, default=True, help='whether condition on time')
    parser.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    parser.add_argument('--extend_radius', type=eval, default=True, help='Extend edges to radius graph')
    parser.add_argument('--normalization_factor', type=float, default=1, help='Normalize the sum aggregation of EGNN')
    parser.add_argument('--aggregation_method', type=str, default='sum', help='"sum" or "mean"')
    parser.add_argument('--edge_order', type=int, default=3)
    parser.add_argument('--cutoff', type=float, default=10.0)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
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

    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    with open('%s/args.pickle' % log_dir, 'wb') as f:
        pickle.dump(args, f)

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set_path = os.path.join(os.path.dirname(config.dataset.train), 'train_data_40k_ps'+str(args.vocab_len)+'_dg.pkl')
    val_set_path = os.path.join(os.path.dirname(config.dataset.train), 'val_data_5k_ps'+str(args.vocab_len)+'_dg.pkl')
    train_set = ConformationDataset(train_set_path, transform=transforms)
    val_set = ConformationDataset(val_set_path, transform=transforms)
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, shuffle=True))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # Model
    logger.info('Building model...')
    model = get_model(config, args).to(args.device)

    # Optimizer
    optimizer = get_optimizer(config.train.optimizer, model)
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = EMA(args.ema_decay)
    else: 
        ema = None
        model_ema = model

    gradnorm_queue = Queue()
    gradnorm_queue.add(30000) 
    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])


    def train(it):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss_a_alg, t_steps = model.get_loss(batch)
        loss = loss_a_alg
        loss.backward()

        grad_norm = gradient_clipping(model, gradnorm_queue)
        optimizer.step()

        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        logger.info('[Train] Iter %05d | Loss %.5f | Grad %.2f | LR %.6f ' % (
            it, loss.item(), grad_norm, optimizer.param_groups[0]['lr']))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', grad_norm, it)
        writer.flush()

    def validate(it):
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch = batch.to(args.device)
                loss_a_alg, t_steps = model.get_loss(batch)
                avg_loss = loss_a_alg.item()
        
        logger.info('[Validate] Iter %05d | Loss %.5f | L_tot %.5f' % (
            it, loss_a_alg.item(), avg_loss))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'model_ema': model_ema.state_dict() if args.ema_decay > 0 else None,
                    'optimizer': optimizer.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

 