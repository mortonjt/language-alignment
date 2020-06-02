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
from language_alignment.alignment import matrix_to_edges, aln2edges
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
    sx = x.replace('-', '')
    sy = y.replace('-', '')
    truth_edges = aln2edges(x, y)
    # Estimate edges
    sx = seq2onehot(sx.upper()).to(device)
    sy = seq2onehot(sy.upper()).to(device)
    dm, _ = model.align(sx, sy)
    pred_edges = matrix_to_edges(dm.cpu().detach().numpy())
    pred_edges = pred_edges.values.tolist()
    res = score_alignment(pred_edges, truth_edges, len(x))
    tp, fp, tn, fn = res

    pred_edge_file = f'{args.output_dir}/edges.csv'
    pd.DataFrame(
        pred_edges
    ).to_csv(pred_edge_file, index=None, header=None)
    pred_stats_file = f'{args.output_dir}/stats.csv'
    stats = pd.Series({'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn})
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
