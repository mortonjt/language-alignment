import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_

from language_alignment.dataset import collate_alignment_pairs
from language_alignment.dataset import AlignmentDataset
from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import MeanAligner, SSAaligner, CCAaligner
from language_alignment.losses import TripletLoss


def aligner_type(args):
    input_dim = args.lm_embed_dim
    embed_dim = args.aligner_embed_dim
    max_len = args.max_len
    device = 'cuda' if args.gpu else 'cpu'
    if args.aligner == 'cca':
        align_fun = CCAaligner(input_dim, embed_dim)
    elif args.aligner == 'ssa':
        align_fun = SSAaligner(input_dim, embed_dim)
    else:
        align_fun = MeanAligner()
    return align_fun


def init_model(args):
    cls, path = pretrained_language_models[args.arch]
    device = 'cuda' if args.gpu else 'cpu'
    if args.lm is not None:
        path = args.lm
    align_fun = aligner_type(args)
    model = AlignmentModel(aligner=align_fun, loss=TripletLoss())
    model.load_language_model(cls, path)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def init_dataloaders(args, device):
    seqs = list((SeqIO.parse(open(args.fasta), format='fasta')))
    seqs = {x.id: x.seq for x in seqs}
    cfxn = collate_alignment_pairs
    test_pairs = pd.read_table(args.test_pairs, header=None,
                               sep='\s+', dtype=str)
    test_dataset = AlignmentDataset(test_pairs, seqs)
    test_dataloader = DataLoader(test_dataset, 1,
                                  shuffle=False, collate_fn=cfxn)
    return test_dataloader, test_pairs
