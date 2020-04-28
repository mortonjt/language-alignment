"""
This file contains 3 different alignment layers

1. SSALayer : Soft symmetric alignment as proposed by Bepler
2. CCALayer : DeepCCA layer
3. RankingLayer : Fixed size differencing representation
"""

import torch
import torch.nn as nn


class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1) - y), -1)


class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1) - y)**2, -1)


class SSALayer(nn.Module):
    def __init__(self, beta_init, compare=L1()):
        self.theta = nn.Parameter(torch.ones(1, n_classes - 1))
        self.beta = nn.Parameter(torch.zeros(n_classes - 1) +
                                 beta_init)
        self.compare = compare

    def __call__(self, z_x, z_y):
        s = self.compare(z_x, z_y)

        a = F.softmax(s, 1)
        b = F.softmax(s, 0)

        a = a + b - a * b
        c = torch.sum(a * s) / torch.sum(a)
        return c


class RankingLayer(nn.Module):
    def __init__(self, input_size, emb_dim):
        """ Initialize model parameters for Siamese network.

        This is another forum of triplet loss.

        Parameters
        ----------
        input_size: int
            Input dimension size
        emb_dim: int
            Embedding dimension for both datasets

        Note
        ----
        This implicitly assumes that the embedding dimension for
        both datasets are the same.
        """
        # See here: https://adoni.github.io/2017/11/08/word2vec-pytorch/
        super(RankingLayer, self).__init__()
        self.input_size = input_size
        self.emb_dimension = emb_dim
        self.output = nn.Linear(input_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.emb_dimension)
        self.output.weight.data.normal_(0, initstd)

    def forward(self, pos, neg):
        """
        Parameters
        ----------
        pos : torch.Tensor
           Positive shared representation vector
        neg : torch.Tensor
           Negative shared representation vector(s).
           There can be multiple negative examples (~5 according to NCE).
        """
        losses = 0
        pos_out = self.output(pos)
        neg_out = self.output(neg)
        diff = pos - neg_out
        #score = F.logsigmoid(diff)
        #losses = sum(score)
        losses = sum(torch.norm(diff))
        return -1 * losses
