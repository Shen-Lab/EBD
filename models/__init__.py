import torch
from .hd import EquivariantHeatDissipation
from .hier_enc import HierarchicalMessagePassing


def get_model(config, args):
    encoder = HierarchicalMessagePassing(args)
    if args.type == 'hd':
        return EquivariantHeatDissipation(config, args, encoder)
    else:
        raise NotImplementedError('Unknown model: %s' % args.type)
