#!/usr/bin/env python


from Bio import SeqIO
import os
import numpy as np
import pandas as pd
import sys
import pickle
from scipy.spatial.distance import euclidean, cosine
from sklearn.cross_decomposition import CCA
import glob
import copy
import networkx as nx
import site
from functools import reduce

#base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#site.addsitedir(os.path.join(base_path, 'src'))

from language_alignment.alignment import cca_solve, cca_align, filter_by_locality


#fasta_file = '../data/combined.fasta'
in_directory = sys.argv[1]
pair_file = sys.argv[2]
results_dir = sys.argv[3]
components = int(sys.argv[4])

# components=3
# in_file = '../data/domains/domain_pairs/domain_pair_aa'
# pair_file = '../data/domains/domain_pairs.txt'
# dom_file = '../data/domains/swissprot-pfam-domains.csv'
# out_file = '../results/pw_attn_domain_alignment_test.txt'
#components = 40

#out_file = f'../results/pw_attention_cca_c{components}.txt'
#out_file = 'PF00005_contacts.txt'
L = 768
N = 1022
results = []

#fnames = glob.glob('../../results/swissprot-attn/*.npz')
fnames = glob.glob(f'{in_directory}/*.npz')
fnames2 = list(map(os.path.basename, fnames))
qs = list(map(lambda x: x.split('.npz')[0], fnames2))
qsd = dict(zip(qs, fnames))

# watch out for this
pairs = pd.read_csv(pair_file, sep='\s+', header=None)
print(pairs.shape, len(qsd))
# with open(out_file, 'w') as out_handle:
for i in range(len(pairs)):

    t = pairs.iloc[i].values.ravel()
    prot_x, prot_y = t[0], t[1]
    outname = f'{results_dir}/attn_edges_{prot_x}{prot_y}.csv'

    # don't reprocess edges if it exists
    if os.path.exists(outname): continue

    if prot_x in qsd and prot_y in qsd:
        data_x = np.load(qsd[prot_x])
        data_y = np.load(qsd[prot_y])
        x = copy.copy(data_x['embed'])
        y = copy.copy(data_y['embed'])
        data_x.close()
        data_y.close()

        try:
            Ux, Vy, phix, psiy, cca_xy = cca_solve(x.T, y.T, components)
            # now obtain the benchmark
            edge_xy = cca_align(phix, psiy)
            edge_xy = filter_by_locality(edge_xy)
            edge_xy.to_csv(outname)

        except Exception as e:
            print(prot_x, prot_y, 'not processed')
            continue
