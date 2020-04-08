from Bio import SeqIO
import torch
import numpy as np
import sys
import pickle
# note: must be in roberta virtualenv
import os
import glob
from seqvec.seqvec import get_elmo_model
from pathlib import Path

in_file = sys.argv[1]
model_path = Path('model')
results_dir = sys.argv[2]

device = 'cuda:0'
#path='data/attn'
path = os.path.dirname(model_path)
model = os.path.basename(model_path)
print(path, model)

model = get_elmo_model(model_path, cpu=False)

data_dir = f'{path}/data-bin'
dict_path = '/mnt/home/mgt/roberta_checkpoints'

L = 768
N = 1022
results = []
precomputed = set(glob.glob(f'{results_dir}/*.npz'))
with open(in_file, "rU") as input_handle:

    for record in SeqIO.parse(input_handle, "fasta"):
        s = str(record.seq)
        if len(s) > N: continue

        fname = f'{results_dir}/{record.id}.npz'
        if fname in precomputed:
            print(f'{fname} is already computed')
        else:
            print(record.id)
            res = model.embed_sentence(list(s))
            embed = res.squeeze()
            np.savez_compressed(fname, embed=embed,
                                sequence=np.array(s))
