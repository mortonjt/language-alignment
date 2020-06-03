import argparse
import pandas as pd
import os
import numpy as np


parser = argparse.ArgumentParser(description='Evaluates edit distance quality.')
parser.add_argument('-p','--test-pairs', help='Testing pairs', required=True)
parser.add_argument('-a','--alignments', help='Table of alignments', required=True)
parser.add_argument('-o','--output-file', help='Output directory of edges', required=True)
args = parser.parse_args()

# Read in files
test_pairs = pd.read_table(args.test_pairs, header=None,
                           sep='\s+', dtype=str)
cols = test_pairs.columns
blast_df = pd.read_table(args.alignments, header=None, dtype=str)

blast_df.columns = [
    'cur.id', 'hit.id', 'i',
    'qs', 'qe', 'he', 'hs',
    'query_s', 'hit_s', 'aln_s',
    'bitscore', 'evalue'
]
blast_df['bitscore'] = blast_df['bitscore'].astype(np.float)
blast_df['evalue'] = blast_df['evalue'].astype(np.float)

blast_df = blast_df.dropna()
blast_df = blast_df.set_index(['cur.id', 'hit.id'])
dist_xy, dist_xz = [], []
for i in range(len(test_pairs)):
    x, y, z = test_pairs.iloc[i]
    if (x, y) in blast_df.index:
        dxy = blast_df.loc[(x, y), 'evalue'].values[0]
    elif (y, x) in blast_df.index:
        dxy = blast_df.loc[(y, x), 'evalue'].values[0]
    else:
        dxy = 100000

    if (x, z) in blast_df.index:
        dxz = blast_df.loc[(x, z), 'evalue'].values[0]
    elif (z, x) in blast_df.index:
        dxz = blast_df.loc[(z, x), 'evalue'].values[0]
    else:
        dxz = 100000

    dist_xy.append(dxy)
    dist_xz.append(dxz)

test_pairs['dxy'] = dist_xy
test_pairs['dxz'] = dist_xz
test_pairs.to_csv(args.output_file, sep='\t', index=None, header=None)
