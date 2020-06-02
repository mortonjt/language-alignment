import torch
from language_alignment.layers import MeanAligner, CCAaligner, SSAaligner
import subprocess


class AlignmentModel(torch.nn.Module):

    def __init__(self, aligner, loss):
        """
        Parameters
        ----------
        aligner : Aligner instance
           Either MeanAligner, CCAaligner or SSAaligner
        loss : TripletLoss instance
        """
        super(AlignmentModel, self).__init__()
        self.aligner_fun = aligner
        self.loss_fun = loss

    def load_language_model(self, cls, path):
        """
        Parameters
        ----------
        cls : Module name
            Name of the Language model.
            (i.e. binding_prediction.language_model.Elmo)
        path : filepath
            Filepath of the pretrained model.
        """
        self.lm = cls(path)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x : torch.Tensor
           Embedding for sequence x
        y : torch.Tensor
           Embedding for sequence y
        """
        z_x = self.lm(x)
        z_y = self.lm(y)
        return self.aligner_fun(z_x, z_y)

    def align(self, x, y, condense=False, max_len=1024):
        p = len(x)
        q = len(y)
        # sx = torch.zeros(max_len).long()
        # sy = torch.zeros(max_len).long()
        # sx[:p] = x
        # sy[:q] = y
        z_x = self.lm(x)
        z_y = self.lm(y)
        dm, score = self.aligner_fun.align(z_x, z_y, condense)
        return dm, score

    def predict(self, z_x, z_y):
        dist = self.forward(z_x, z_y)
        return dist

    def loss(self, x, y, z):
        h_x = self.lm(x)
        h_y = self.lm(y)
        h_z = self.lm(z)
        xy = self.aligner_fun(h_x, h_y)
        xz = self.aligner_fun(h_x, h_z)
        l = self.loss_fun(xy, xz)
        return l
