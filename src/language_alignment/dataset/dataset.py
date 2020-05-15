r""" This implements the datsets from Bepler et al 2019."""
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from language_alignment.util import check_random_state


def seq2onehot(seq):
    """ Create 21-dim 1-hot embedding """
    # chars = ['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y',
    #          'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E', '-']
    # vocab_size = len(chars)
    # vocab_embed = dict(zip(chars, range(vocab_size)))
    vocab_embed = {
        '<start>': 0, 'A': 5, 'B': 25, 'C': 22, 'D': 13,
        'E': 9, 'F': 18, 'G': 7, 'H': 20, 'I': 12, 'J': 3,
        'K': 14, 'L': 4, 'M': 21, 'N': 16, 'O': 28, 'P': 15,
        'Q': 17, 'R': 10, 'S': 6, 'T': 11, 'U': 26, 'V': 8,
        'W': 23, 'X': 24, 'Y': 19, 'Z': 27, '.': 3, '<end>': 2}
    vocab_size = len(vocab_embed)
    # Convert vocab to one-hot
    vocab_one_hot = np.eye(vocab_size)
    seqs_x = [vocab_embed[v] for v in seq]
    #seqs_x = [vocab_one_hot[j, :] for j in embed_x]
    # ignore the ends for now
    #seqs_x = [vocab_embed['<start>']] + seqs_x + [vocab_embed['<end>']]
    return torch.Tensor(np.array(seqs_x)).long()

def collate_alignment_pairs(batch, device, max_len=1024, pad=0):
    """
    Padds matrices of variable length
    """
    # get sequence lengths
    lengths = torch.tensor(
        [(t[0].shape[0], t[1].shape[0], t[2].shape[0])
             for t in batch])
    lengths = lengths.to(device)
    ml = lengths.max()
    S1_padded = torch.zeros((len(batch), ml))
    S2_padded = torch.zeros((len(batch), ml))
    S3_padded = torch.zeros((len(batch), ml))
    # Is this padding correct??
    S1_padded[:, :] = pad
    S2_padded[:, :] = pad
    S3_padded[:, :] = pad

    # padd (double check for dim issues)
    for i in range(len(batch)):
        S1_padded[i][:lengths[i][0]] = batch[i][0]
        S2_padded[i][:lengths[i][1]] = batch[i][1]
        S3_padded[i][:lengths[i][2]] = batch[i][2]

    if S1_padded.shape[1] > max_len:
        S1_padded = S1_padded[:, :max_len]
    if S2_padded.shape[1] > max_len:
        S2_padded = S2_padded[:, :max_len]
    if S3_padded.shape[1] > max_len:
        S3_padded = S3_padded[:, :max_len]

    s1 = S1_padded.long().to(device)
    s2 = S2_padded.long().to(device)
    s3 = S3_padded.long().to(device)
    return s1, s2, s3


def random_swap(x):
    """ Data augmentation. """
    alpha = list('RXSGWIQATVKYCNLFDMPHE')
    X = list(x)
    a = np.random.randint(len(alpha))
    i = np.random.randint(len(X))
    X[i] = alpha[a]
    return X


class AlignmentDataset(Dataset):
    """ Dataset for training and testing. """
    def __init__(self, pairs, seqs, tokenizer=seq2onehot):
        """ Read in pairs of proteins

        Parameters
        ----------
        pairs: np.array of str
            Pairs of proteins that align.
        sampler : NegativeSampler
            Model for drawing negative samples for training
        sort : bool
            Specifies if the pairs should be sorted by
            protein id1 then by taxonomy.
        seed : int
            Random seed
        """
        self.pairs = pairs
        self.seqs = seqs
        self.tokenizer = tokenizer

    def random_peptide(self):
        if self.sampler is None:
            raise ("No negative sampler specified")
        return self.sampler.draw()

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, i):
        """
        Parameters
        ----------
        i : int
           Index of item

        Returns
        -------
        gene : torch.Tensor
           Encoded representation of protein of interest
        pos : torch.Tensor
           Encoded representation of protein that interacts with `gene`.
        neg : torch.Tensor
           Encoded representation of protein that probably doesn't
           interact with `gene`.
        """
        gene = self.pairs.loc[i, 0]
        pos = self.pairs.loc[i, 1]
        neg = self.pairs.loc[i, 2]
        gene = random_swap(str(self.seqs[gene]))
        pos = random_swap(str(self.seqs[pos]))
        neg = random_swap(str(self.seqs[neg]))
        gene = self.tokenizer(gene).long()
        pos = self.tokenizer(pos).long()
        neg = self.tokenizer(neg).long()
        return gene, pos, neg

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.pairs)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                yield self.__getitem__(i)


class MultinomialResample:
    def __init__(self, trans, p):
        self.p = (1-p)*torch.eye(trans.size(0)).to(trans.device) + p*trans

    def __call__(self, x):
        #print(x.size(), x.dtype)
        p = self.p[x] # get distribution for each x
        return torch.multinomial(p, 1).view(-1) # sample from distribution


class SubstituteSwap(object):
    """ Base pair substitution"""
    def __init__(self, p, alphabet):

        pass

    def __call(self, x):
        pass
