from Bio import SeqIO
import torch
import numpy as np
import sys
import pickle
# note: must be in roberta virtualenv
import os
import glob
from transformers import *
import warnings
warnings.filterwarnings("ignore")

in_file = sys.argv[1]
model_path = sys.argv[2]
results_dir = sys.argv[3]

device = 'cuda:0'
#path='data/attn'
path = os.path.dirname(model_path)
model = os.path.basename(model_path)
print(path, model)

#bert_dir = '/mnt/home/jmorton/ceph/models/Bert/Uniref100'
model = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer(f"{model_path}/vocab.txt",
                          do_lower_case=False)
model.to(device)
for p in model.parameters():
    p.requires_grad = False

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
            tokens = tokenizer.encode(' '.join(list(s)),
                                      add_special_tokens=True)
            tokens = torch.tensor(tokens).view(-1, 1)
            tokens = tokens.to(device)
            res = model(tokens)[0]
            embed = res.detach().cpu().numpy().squeeze()
            np.savez_compressed(fname, embed=embed,
                                sequence=np.array(s))

            del tokens
