import sys
from scipy.spatial.distance import euclidean, squareform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import os
import re
import site
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(os.path.join(base_path, 'src'))
from score import domain_score
from alignment import cca_solve


embed_directory = sys.argv[1]
msa_metadata = sys.argv[2]
out_file = sys.argv[3]
transpose = bool(sys.argv[4])
mode = sys.argv[5]
if mode is None:
    mode = 'euclidean'
if transpose is None:
    transpose = False

def distance(x, y):
    if transpose:
        X = x.T
        Y = y.T

    if mode == 'euclidean':
        xc = X.mean(axis=0)
        yc = Y.mean(axis=0)
        return euclidean(xc, yc)

    elif mode == 'cca':
        r2 = cca_solve(X, Y)[-1]
        return 1 - r2


def get_distances(path, names):
    dists = []
    for j in range(len(names)):
        for i in range(j):

            try:
                ni = os.path.join(path, names[i])
                nj = os.path.join(path, names[j])
                x = np.load(ni)['embed']
                y = np.load(nj)['embed']
                r = (names[i], names[j], distance(x, y))
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
    for w in list(within_names.values):
        for o in list(outside_names.values):
            dw = np.unique(dm.loc[tuple(w)].distance)
            do = np.unique(dm.loc[tuple(o)].distance)
            c+= int(dw < do)
    return c / (len(within_names) * len(outside_names))


metadata = pd.read_table(msa_metadata, index_col=0, sep='\s+')

# obtain embedding files
files = glob.glob(embed_directory + '/*')
files = list(map(lambda x: os.path.basename(x), files))
print('data loaded')
dm = {}
# calculate distances
for name, group in metadata.groupby('family'):
    print(name)
    within = list(group.loc[group['within'], 'from'])
    outside = list(group.loc[group['outside'], 'to'])
    kf = get_distances(embed_directory, names=within + outside)
    kf['within'] = kf.apply(lambda x: x['from'] in within and x['to'] in within, axis=1)
    kf['outside'] = kf.apply(lambda x: x['from'] in within and x['to'] in outside, axis=1)
    dm[name] = count(kf)
dm = pd.Series(dm, name='dist')
dm.to_csv(out_file, sep='\t')
