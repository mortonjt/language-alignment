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

from language_model.model_utils import init_model, init_dataloaders


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
