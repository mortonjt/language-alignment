"""
This file contains 3 different alignment layers

1. SSALayer : Soft symmetric alignment as proposed by Bepler
2. CCALayer : DeepCCA layer
3. RankingLayer : Fixed size differencing representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
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
    def __init__(self, input_dim=512, embed_dim=64, compare=L1()):
        super(SSAaligner, self).__init__()
        self.compare = compare
        self.projection = nn.Linear(input_dim, embed_dim)
        torch.nn.init.xavier_uniform(self.projection.weight)
        self.projection.bias.data.fill_(0.1)

    def __call__(self, z_x, z_y):
        z_x = z_x.permute(0, 2, 1).contiguous()
        z_y = z_y.permute(0, 2, 1).contiguous()
        x = self.projection(z_x)
        y = self.projection(z_y)
        x = x.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()

        s = self.compare(x, y)

        a = F.softmax(s, 1)
        b = F.softmax(s, 0)

        a = a + b - a * b
        c = torch.sum(a * s) / torch.sum(a)
        return c


class CCAaligner(nn.Module):
    def __init__(self, input_dim=512, embed_dim=64, device='cpu'):
        super(CCAaligner, self).__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        #self.batch_norm = nn.BatchNorm1d(embed_dim, embed_dim)
        self.loss = CCAloss(embed_dim, device=device)
        torch.nn.init.xavier_uniform(self.projection.weight)
        self.projection.bias.data.fill_(0.1)

    def __call__(self, z_x, z_y):
        """
        Notes
        -----
        Assumes that the first dimension of z_x and z_y
        corresponds to the sequence length and the
        second dimensions corresponds to the embedding dimension.

        The input *must* be padded. Only accepts 1 pair at a time.
        """

        # transpose for projection
        z_x = z_x.permute(0, 2, 1).contiguous()
        z_y = z_y.permute(0, 2, 1).contiguous()
        x = self.projection(z_x)
        y = self.projection(z_y)
        x = x.permute(0, 2, 1).contiguous().double()
        y = y.permute(0, 2, 1).contiguous().double()

        return self.loss(x, y)
