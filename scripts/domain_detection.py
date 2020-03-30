#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run domain detection
"""
import os
import logging
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import euclidean, cosine
import sys

in_directory = sys.argv[1]
triples_file = sys.argv[2]
out_file = sys.argv[3]

triplets = pd.read_table(triples_file, index_col=None)

embed_files = glob.glob(f'{in_directory}/*.npz')
embed_base = list(map(os.path.basename, embed_files))
embed_base = list(map(lambda x: x.split('.')[0], embed_base))
acc2emb = dict(zip(embed_base, embed_files))

dfs = []
columns = [*triplets.columns, 'euclidean_r1r2', 'euclidean_r1r3', 'cosine_r1r2', 'cosine_r1r3']
tot  = len(triplets)
skip = tot // 1000

with open(out_file, 'w') as outfile:
    #print(*columns, sep='\t', file=outfile)

    for i, (_, row) in enumerate(triplets.iterrows()):
        if row.protein not in acc2emb: continue

        fname = acc2emb[row.protein]
        data = np.load(fname)
        emb = data['embed']
        data.close()
        r1, r2, r3 = map(lambda x: x - 1, row[['r1','r2','r3']])  # 0-index it
        er1, er2, er3 = emb[r1], emb[r2], emb[r3]

        row['euclidean_r1r2'] = euclidean(er1, er2)
        row['euclidean_r1r3'] = euclidean(er1, er3)
        row['cosine_r1r2']    = cosine(er1, er2)
        row['cosine_r1r3']    = cosine(er1, er3)
        print(*[row[col] for col in columns], sep='\t', file=outfile)
        if not i % skip and i or not i % tot:
            print(f"{i}/{len(triplets)} comparisons made.")
