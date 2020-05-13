import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import datetime
import glob
import os
import re
from Bio import SeqIO
from tqdm import tqdm
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
        align_fun = CCAaligner(input_dim, embed_dim, device=device)
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
    model = AlignmentModel(aligner=align_fun)
    model.load_language_model(cls, path, device=device)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def init_dataloaders(args, device):
    seqs = list((SeqIO.parse(open(args.fasta), format='fasta')))
    seqs = {x.id: x.seq for x in seqs}
    cfxn = lambda x: collate_alignment_pairs(x, device)
    test_pairs = pd.read_table(args.test_pairs, header=None,
                               sep='\s+', dtype=str)
    test_dataset = AlignmentDataset(test_pairs, seqs)
    test_dataloader = DataLoader(test_dataset, 1,
                                  shuffle=False, collate_fn=cfxn)
    return test_dataloader, test_pairs


def main(args):
    # set seed for debugging
    # torch.manual_seed(0)
    # Initialize model
    model, device = init_model(args)
    # Read in all data
    test_dataloader, test_pairs = init_dataloaders(args, device)
    # Estimate distances
    vals = []
    for batch in tqdm(test_dataloader):
        x, y, z = batch[0], batch[1], batch[2]
        xy = model.predict(x, y).item()
        xz = model.predict(x, z).item()
        vals.append((xy, xz))
    res = pd.DataFrame(vals)
    res = pd.concat((test_pairs, res), axis=1)

    # Write results to file
    res.to_csv(args.output_file, sep='\t', index=None, header=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--test-pairs', help='Validation pairs file', required=True)
    parser.add_argument('--fasta', help='Fasta file', required=True)
    parser.add_argument('-m','--lm', help='Path to trained alignment model.',
                        required=False, default=None)
    parser.add_argument('-c','--arch',
                        help='Pretrained model type (choices include onehot, elmo and roberta',
                        required=False, default='elmo')
    parser.add_argument('-a','--aligner',
                        help='Aligner type. Choices include (mean, cca, ssa).',
                        required=False, type=str, default='mean')
    parser.add_argument('--model-path', help='Model path',
                        required=False, default=None)
    parser.add_argument('--lm-embed-dim', help='Language model embedding dimension.',
                        required=False, type=int, default=1024)
    parser.add_argument('--aligner-embed-dim', help='Aligner embedding dimension.',
                        required=False, type=int, default=128)
    parser.add_argument('--max-len', help='Maximum length of protein', default=1024,
                        required=False, type=str)
    parser.add_argument('-g','--gpu', help='Use GPU or not', default=False,
                        required=False, type=bool)
    parser.add_argument('-o','--output-file', help='Output file of model results',
                        required=True)
    args = parser.parse_args()
    main(args)
