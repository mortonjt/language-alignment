import sys
from scipy.spatial.distance import euclidean, squareform
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import os
import re
import site
#base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#site.addsitedir(os.path.join(base_path, 'src'))
#from score import domain_score
from language_alignment.alignment import cca_solve
from language_alignment.score import distance



parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--input-directory', help='Input directory', required=True)
parser.add_argument('-m','--metadata', help='Triples', required=True)
parser.add_argument('-r','--transpose', help='Transpose embeddings',
                    required=False, default=False, type=bool)
parser.add_argument('-d','--distance', help='Distance metric',
                    required=False, default='cca', type=str)
parser.add_argument('-o','--out_file', help='Output directory of edges', required=True)
args = parser.parse_args()



embed_directory = args.input_directory
msa_metadata = args.metadata
out_file = args.out_file
mode = args.distance
transpose = args.transpose

#TODO fix above

# transpose = bool(sys.argv[4])
# mode = sys.argv[5]
# if mode is None:
#     mode = 'euclidean'
# if transpose is None:
#     transpose = False

def get_distances(path, names):
    dists = []
    for j in range(len(names)):
        for i in range(j):

            try:
                ni = os.path.join(path, names[i])
                nj = os.path.join(path, names[j])
                x = np.load(ni)['embed']
                y = np.load(nj)['embed']
                r = (names[i], names[j], distance(x, y, mode, transpose))
                dists.append(r)
            except:
                r = (names[i], names[j], 100)
                dists.append(r)
    #dm = squareform(dists)
    #dm = pd.DataFrame(dm, index=names, columns=names)
    #dm = dm.fillna(0)
    return pd.DataFrame(dists, columns=['from', 'to', 'distance'])


def count(dm):
    within_names = dm.loc[dm.within, ['from', 'to']]
    outside_names = dm.loc[dm.outside, ['from', 'to']]
    dm = dm.set_index(['from', 'to'])
    idx = list(zip(within_names, outside_names))
    c = 0
    total = 0
    for w in list(within_names.values):
        for o in list(outside_names.values):
            dw = np.unique(dm.loc[tuple(w)].distance)
            do = np.unique(dm.loc[tuple(o)].distance)
            c+= int(dw < do)
            total += 1
    return c / total


metadata = pd.read_table(msa_metadata, index_col=0, sep='\s+')

# obtain embedding files
files = glob.glob(embed_directory + '/*')
files = list(map(lambda x: os.path.basename(x), files))
print('data loaded')
dm = {}
# calculate distances
for name, group in metadata.groupby('family'):
    print(name)
    within = list(group.loc[group['within'], 'from'].unique())
    outside = list(group.loc[group['outside'], 'to'].unique())
    if len(within) <= 2: continue
    kf = get_distances(embed_directory, names=within + outside)
    kf['within'] = kf.apply(lambda x: x['from'] in within and x['to'] in within, axis=1)
    kf['outside'] = kf.apply(lambda x: x['from'] in within and x['to'] in outside, axis=1)
    dm[name] = count(kf)
dm = pd.Series(dm, name='dist')
dm.to_csv(out_file, sep='\t')
