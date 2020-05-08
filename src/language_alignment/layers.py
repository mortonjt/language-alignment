"""
This file contains 3 different alignment layers

1. SSALayer : Soft symmetric alignment as proposed by Bepler
2. CCALayer : DeepCCA layer
3. RankingLayer : Fixed size differencing representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from language_alignment.losses import CCAloss


class MeanAligner(nn.Module):
    def __call__(self, z_x, z_y):
        mean_x = z_x.mean(axis=-1)
        mean_y = z_y.mean(axis=-1)
        return (mean_x - mean_y).pow(2).sum()


class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1) - y), -1)


class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1) - y)**2, -1)


class SSAaligner(nn.Module):
    def __init__(self, compare=L1()):
        super(SSAaligner, self).__init__()
        self.compare = compare

    def __call__(self, z_x, z_y):
        x = torch.squeeze(z_x).t()
        y = torch.squeeze(z_y).t()
        s = self.compare(x, y)

        a = F.softmax(s, 1)
        b = F.softmax(s, 0)

        a = a + b - a * b
        c = torch.sum(a * s) / torch.sum(a)
        return c


class CCAaligner(nn.Module):
    def __init__(self, input_dim=512, embed_dim=64, max_len=1024, device='cpu'):
        super(CCAaligner, self).__init__()
        self.model_x = nn.Linear(input_dim, embed_dim)
        self.model_y = nn.Linear(input_dim, embed_dim)
        self.loss = CCAloss(embed_dim, device=device)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_len = max_len

    def __call__(self, z_x, z_y):
        """
        Notes
        -----
        Assumes that the first dimension of z_x and z_y
        corresponds to the sequence length and the
        second dimensions corresponds to the embedding dimension.

        The input *must* be padded. Only accepts 1 pair at a time.
        """
        x = torch.squeeze(z_x).t()
        y = torch.squeeze(z_y).t()
        x = self.model_x(x)
        y = self.model_y(y)
        return self.loss(x, y)
