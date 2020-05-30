import argparse
import pandas as pd
import os
import numpy as np
from Bio import SeqIO
from language_alignment.smith_waterman import init_aligner, pairwise_align


parser = argparse.ArgumentParser(description='Evaluates edit distance quality.')
parser.add_argument('-f','--fasta', help='Fasta file of sequences', required=True)
parser.add_argument('-p','--test-pairs', help='Testing pairs', required=True)
parser.add_argument('-o','--output-file', help='Output directory of edges', required=True)
args = parser.parse_args()

# Read in Fasta file
handle = SeqIO.parse(open(args.fasta), format='fasta')
seqdict = {x.id : str(x.seq) for x in handle}

# Read in pairs
test_pairs = pd.read_table(args.test_pairs, header=None,
                           sep='\s+', dtype=str)

dist_xy, dist_xz = [], []
# Run smith-waterman
aligner = init_aligner()
for i in range(len(test_pairs)):
    x, y, z = test_pairs.iloc[i]
    sx, sy, sz = seqdict[x], seqdict[y], seqdict[z]
    dxy, _ = pairwise_align(aligner, sx, sy)
    dxz, _ = pairwise_align(aligner, sx, sz)

    dist_xy.append(dxy)
    dist_xz.append(dxz)

test_pairs['dxy'] = dist_xy
test_pairs['dxz'] = dist_xz
test_pairs.to_csv(args.output_file, sep='\t', index=None, header=None)
