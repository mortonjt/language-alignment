from vocabulary import Vocab
import torch
from mem_transformer import MemTransformerLM

vocabFile = 'vocab.txt'
modelChkFile = '/mnt/home/jmorton/ceph/models/TransformerXL/Uniref100/model.pt'
device = torch.device('cuda')

vocab = Vocab(lower_case=False,special=['<S>'])
vocab.count_file(vocabFile)
vocab.build_vocab()

encoded_data = list()

#This is a list of all your samples
data = ['L A E C A A T A', 'M A E C A A T X', 'A A E C A A T']

for sample in data:
    symbols = vocab.tokenize(sample, add_eos=False,
                        add_double_eos=False)
    encoded_data.append(vocab.convert_to_tensor(symbols))


## BFD Model
# model = MemTransformerLM(n_token=22, n_layer=32, n_head=14, d_model=1024,
#                              d_head=128, d_inner=4096, dropout=0.0, dropatt=0.0,
#                              tie_weight=True, d_embed=1024, div_val=1,
#                              tie_projs=[False], pre_lnorm=False, tgt_len=512,
#                              ext_len=0, mem_len=512, cutoffs=[],
#                              same_length=False, attn_type=0,
#                              clamp_len=-1, sample_softmax=-1)

## Uniref Model
model = MemTransformerLM(n_token=22, n_layer=30, n_head=16, d_model=1024,
                             d_head=64, d_inner=4096, dropout=0.0, dropatt=0.0,
                             tie_weight=True, d_embed=1024, div_val=1,
                             tie_projs=[False], pre_lnorm=False, tgt_len=512,
                             ext_len=0, mem_len=512, cutoffs=[],
                             same_length=False, attn_type=0,
                             clamp_len=-1, sample_softmax=-1)


model.to(device);

state_dict = torch.load(modelChkFile, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)

model.eval();

embedding = list()

for encoded_sample in encoded_data:
    with torch.no_grad():
        # the original code for Transformer-XL used shapes [len, bsz]\n",
        # encoded_data = encoded_sample.unsqueeze(1)
        encoded_data = encoded_sample.unsqueeze(1).cuda()
        #Predict hidden states features for each layer
        hidden_states_1, mems_1 = model(encoded_data)
        #We can re-use the memory cells in a subsequent call to attend a longer context\n",
        #hidden_states_2, mems_2 = model(torch.LongTensor(encoded_data[1]).to(device), *mems_1)
        embedding.append(hidden_states_1)


print(embedding)
