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
import torch
from Bio import SeqIO
from tqdm import tqdm
from language_alignment.model_utils import init_model, init_dataloaders
from language_alignment.alignment import (
    matrix_to_edges, aln2edges, filter_by_locality)
from language_alignment.score import score_alignment
from language_alignment.dataset import seq2onehot


def main(args):
    # set seed for debugging
    # torch.manual_seed(0)
    # Initialize model
    model, device = init_model(args)

    text = open(args.manual_alignment).read()
    x, y = text.split('\n')[:2]
    x, y = x.rsplit()[0], y.rsplit()[0]

    # Obtain ground truth edges
    truth_edges = aln2edges(x, y)
    x = x.replace('-', '')
    y = y.replace('-', '')

    # Estimate edges
    sx = model.lm.model.encode(' '.join(list(x.upper()))).to(device)
    sy = model.lm.model.encode(' '.join(list(y.upper()))).to(device)
    dm, corr = model.align(sx, sy, condense=True)
    dm = dm.cpu().detach().numpy()
    pred_edges = matrix_to_edges(-dm)

    #pred_edges = filter_by_locality(pred_edges)
    pred_edges = pred_edges.astype(np.int64)
    pred_edges = pred_edges[['source', 'target']].values.tolist()
    pred_edges = list(map(tuple, pred_edges))

    res = score_alignment(pred_edges, truth_edges, len(x))
    tp, fp, tn, fn = res

    aln_file = f'{args.output_dir}/alignment_matrix.csv'
    dm = pd.DataFrame(dm, index=list(x), columns=list(y))
    dm.to_csv(aln_file)

    pred_edge_file = f'{args.output_dir}/edges.csv'
    pd.DataFrame(
        pred_edges
    ).to_csv(pred_edge_file, index=None, header=None)
    pred_stats_file = f'{args.output_dir}/stats.csv'
    stats = pd.Series({'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn})
    print(stats)
    stats.to_csv(pred_stats_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates structural alignment.')
    parser.add_argument('--manual-alignment', help='File path to manual alignment',
                        required=True)
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
    parser.add_argument('-o','--output-dir', help='Output directory for model results',
                        required=True)
    args = parser.parse_args()
    main(args)
