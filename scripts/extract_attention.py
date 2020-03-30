from Bio import SeqIO
import torch
import numpy as np
import sys
import pickle
# note: must be in roberta virtualenv
from fairseq.models.roberta import RobertaModel
import os
import glob

in_file = sys.argv[1]
model_path = sys.argv[2]
results_dir = sys.argv[3]

device = 'cuda:0'
#path='data/attn'
path = os.path.dirname(model_path)
model = os.path.basename(model_path)
print(path, model)
data_dir = f'{path}/data-bin'
dict_path = '/mnt/home/mgt/roberta_checkpoints'
roberta = RobertaModel.from_pretrained(
    path, model, data_dir,
    gpt2_encoder_json=f'{path}/peptide_bpe/encoder.json',
    gpt2_vocab_bpe=f'{path}/peptide_bpe/vocab.bpe')
roberta.to(device)

for p in roberta.parameters():
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
            tokens = roberta.encode(' '.join(list(s)))
            tokens = tokens.to(device)
            res = roberta.extract_features(tokens)
            embed = res.detach().cpu().numpy().squeeze()
            np.savez_compressed(fname, embed=embed,
                                sequence=np.array(s))

            del tokens
