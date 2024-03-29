import os
import torch
import numpy as np
import pandas as pd
import argparse
import sys
import copy
import glob
from scipy.spatial.distance import euclidean, cosine
#from statsmodels.multivariate.cancorr import CanCorr
from language_alignment.score import distance
# must load the roberta virtualenv.  It is tweaked
# to handle peptide sequences.
import warnings
warnings.simplefilter("ignore")

# in_directory = sys.argv[1]
# triples_file = sys.argv[2]
# out_file = sys.argv[3]

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--input-directory', help='Input directory', required=True)
parser.add_argument('-t','--triples', help='Triples', required=True)
parser.add_argument('-r','--transpose', help='Transpose embeddings',
                    required=False, default=False, type=bool)
parser.add_argument('-e','--elmo', help='Elmo specified requiements',
                    required=False, default=False, type=bool)
parser.add_argument('-o','--out_file', help='Output directory of edges', required=True)
args = parser.parse_args()

in_directory = args.input_directory
triples_file = args.triples
elmo = args.elmo
out_file = args.out_file
transpose = args.transpose

# in_directory = sys.argv[1]
# triples_file = sys.argv[2]
# out_file = sys.argv[3]
#if len(sys.argv) >= 4:
#    elmo = bool(sys.argv[4])

L = 768
N = 1022
results = []

triples = pd.read_csv(triples_file, sep='\s+', header=None)

fnames = glob.glob(f'{in_directory}/*.npz')
fnames2 = list(map(os.path.basename, fnames))
qs = list(map(lambda x: x.split('.npz')[0], fnames2))
qsd = dict(zip(qs, fnames))
with open(out_file, 'w') as output_handle:
    for i in range(len(triples)):
        t = triples.iloc[i].values.ravel()
        prot_x, prot_y, prot_z = str(t[0]), str(t[1]), str(t[2])
        if i % 1000 == 0:
            print(i, t)
            print(prot_x in qsd, prot_y in qsd, prot_z in qsd)

        if prot_x in qsd and prot_y in qsd and prot_z in qsd:
            try:
                data_x = np.load(qsd[prot_x])
                data_y = np.load(qsd[prot_y])
                data_z = np.load(qsd[prot_z])
                x = copy.copy(data_x['embed'])
                y = copy.copy(data_y['embed'])
                z = copy.copy(data_z['embed'])
                sx = str(copy.copy(data_x['sequence']))
                sy = str(copy.copy(data_y['sequence']))
                sz = str(copy.copy(data_z['sequence']))
                data_x.close()
                data_y.close()
                data_z.close()
            except:
                print(r'Could not parse {prot_x} {prot_y} {prot_z}')
                continue

            if elmo:
                x_ = np.mean(x, axis=1).ravel()
                y_ = np.mean(y, axis=1).ravel()
                z_ = np.mean(z, axis=1).ravel()
            elif len(sx) == x.shape[0]:
                x_ = np.mean(x, axis=0)
                y_ = np.mean(y, axis=0)
                z_ = np.mean(z, axis=0)
            else:
                x_ = np.mean(x[1:-1, :], axis=0)
                y_ = np.mean(y[1:-1, :], axis=0)
                z_ = np.mean(z[1:-1, :], axis=0)

            # Euclidean
            dexy = euclidean(x_, y_)
            dexz = euclidean(x_, z_)
            # CCA
            dcxy = distance(x, y, mode='cca', transpose=transpose)
            dcxz = distance(x, z, mode='cca', transpose=transpose)

            output_handle.write(f'{prot_x}\t{prot_y}\t{prot_z}\t'
                                f'{dexy}\t{dexz}\t{dcxy}\t{dcxz}\n')
