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
from language_alignment.train import LightningAligner


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
    sx = model.model.lm.model.encode(' '.join(list(x.upper()))).to(device)
    sy = model.model.lm.model.encode(' '.join(list(y.upper()))).to(device)
    dm, corr = model.model.align(sx, sy, condense=True)
    dm = dm.cpu().detach().numpy()
    pred_edges = matrix_to_edges(dm)

    #pred_edges = filter_by_locality(pred_edges)
    pred_edges = pred_edges.astype(np.int64)
    pred_edges = pred_edges[['source', 'target']].values.tolist()
    pred_edges = list(map(tuple, pred_edges))

    res = score_alignment(pred_edges, truth_edges, len(x))
    tp, fp, tn, fn = res

    aln_file = f'{args.output_directory}/alignment_matrix.csv'
    dm = pd.DataFrame(dm, index=list(x), columns=list(y))
    dm.to_csv(aln_file)

    truth_edge_file = f'{args.output_directory}/truth_edges.csv'
    pd.DataFrame(
        truth_edges
    ).to_csv(truth_edge_file, index=None, header=None)

    pred_edge_file = f'{args.output_directory}/edges.csv'
    pd.DataFrame(
        pred_edges
    ).to_csv(pred_edge_file, index=None, header=None)
    pred_stats_file = f'{args.output_directory}/stats.csv'
    stats = pd.Series({'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn})
    print(stats)
    stats.to_csv(pred_stats_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates structural alignment.',
                                     add_help=False)
    parser.add_argument('--manual-alignment', help='File path to manual alignment',
                        required=True)
    parser.add_argument('--model-path', help='Model path',
                        required=False, default=None)
    parser = LightningAligner.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
