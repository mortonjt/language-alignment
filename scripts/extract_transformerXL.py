from Bio import SeqIO
import torch
from transformers import *
import numpy as np
import sys
import pickle
import site
# note: must be in roberta virtualenv

import os
import glob
import warnings
warnings.filterwarnings("ignore")


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(os.path.join(base_path, 'src'))

from xl_inference.mem_transformer import MemTransformerLM
from xl_inference.vocabulary import Vocab

in_file = sys.argv[1]
model_path = sys.argv[2]
results_dir = sys.argv[3]

modelChkFile = model_path

device = 'cuda:0'
#path='data/attn'
path = os.path.dirname(model_path)
model = os.path.basename(model_path)
print(path, model)

# TODO: its not great to have a hard-coded path
vocabFile = ('/mnt/home/jmorton/research/gert/icml2020/'
             'language-alignment/src/xl_inference/vocab.txt')
vocab = Vocab(lower_case=False,special=['<S>'])
vocab.count_file(vocabFile)
vocab.build_vocab()

if 'Uniref' in model_path:
    model = MemTransformerLM(
        n_token=22, n_layer=30, n_head=16, d_model=1024,
        d_head=64, d_inner=4096, dropout=0.0, dropatt=0.0,
        tie_weight=True, d_embed=1024, div_val=1,
        tie_projs=[False], pre_lnorm=False, tgt_len=512,
        ext_len=0, mem_len=512, cutoffs=[],
        same_length=False, attn_type=0,
        clamp_len=-1, sample_softmax=-1)
elif 'BFD' in model_path:
     model = MemTransformerLM(
         n_token=22, n_layer=32, n_head=14, d_model=1024,
         d_head=128, d_inner=4096, dropout=0.0, dropatt=0.0,
         tie_weight=True, d_embed=1024, div_val=1,
         tie_projs=[False], pre_lnorm=False, tgt_len=512,
         ext_len=0, mem_len=512, cutoffs=[],
         same_length=False, attn_type=0,
         clamp_len=-1, sample_softmax=-1)
else:
    raise ValueError('Model is misspecified!')

model.to(device)

state_dict = torch.load(modelChkFile, map_location=lambda storage,
                        loc: storage)
model.load_state_dict(state_dict)
model.eval()


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
            try:
                print(record.id)
                tokens = vocab.tokenize(' '.join(list(s)), add_eos=False,
                                        add_double_eos=False)
                tokens = vocab.convert_to_tensor(tokens)
                encoded_data = tokens.unsqueeze(1).cuda()
                encoded_data = encoded_data.to(device)
                res, mems = model(encoded_data)

                embed = res.detach().cpu().numpy().squeeze()
                np.savez_compressed(fname, embed=embed,
                                    sequence=np.array(s))

                del tokens
            except:
                print(record.id, ' had some problems.')
                continue
