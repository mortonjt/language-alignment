""" This implements the datsets from Bepler et al 2019."""
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from language_alignment.util import check_random_state


def collate_alignment_pairs(batch, device):
    """
    Padds matrices of variable length
    """
    # get sequence lengths
    lengths = torch.tensor(
        [max(t[0].shape[0], t[1].shape[0], t[2].shape[0])
             for t in batch]).to(device)
    max_len = max(lengths)
    S1_padded = torch.zeros((len(batch), max_len))
    S2_padded = torch.zeros((len(batch), max_len))
    S3_padded = torch.zeros((len(batch), max_len))
    S1_padded[:, :] = 1
    S2_padded[:, :] = 1
    S3_padded[:, :] = 1

    # padd (double check for dim issues)
    for i in range(len(batch)):
        S1_padded[i][:lengths[i]] = batch[i][0]
        S2_padded[i][:lengths[i]] = batch[i][1]
        S3_padded[i][:lengths[i]] = batch[i][2]

    return (A_padded, S_padded)


class NegativeSampler(object):
    """ Sampler for negative data """
    def __init__(self, seqs):
        self.seqs = seqs

    def draw(self):
        """ Draw at random. """
        i = np.random.randint(0, len(self.seqs))
        return self.seqs[i].seq


class AlignmentDataset(Dataset):
    """ Dataset for training and testing. """
    def __init__(self, pairs, sampler=None, num_neg=10, seed=0):
        """ Read in pairs of proteins

        Parameters
        ----------
        pairs: np.array of str
            Pairs of proteins that align.
        sampler : NegativeSampler
            Model for drawing negative samples for training
        num_neg : int
            Number of negative samples
        sort : bool
            Specifies if the pairs should be sorted by
            protein id1 then by taxonomy.
        seed : int
            Random seed
        """
        self.pairs = pairs
        self.num_neg = num_neg
        self.state = check_random_state(seed)
        self.sampler = sampler
        if sampler is None:
            self.num_neg = 1

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
        gene = self.pairs[i, 0]
        pos = self.pairs[i, 1]
        neg = self.random_peptide()
        gene = ''.join(gene)
        pos = ''.join(pos)
        neg = ''.join(neg)
        return gene, pos, neg

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.pairs)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                for _ in range(self.num_neg):
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
                for _ in range(self.num_neg):
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
