import torch
from language_alignment.layers import MeanAligner, CCAaligner, SSAaligner


class AlignmentModel(torch.nn.Module):

    def __init__(self, aligner):
        """
        """
        super(AlignmentModel, self).__init__()
        self.aligner_fun = aligner

    def load_language_model(self, cls, path, device='cuda'):
        """
        Parameters
        ----------
        cls : Module name
            Name of the Language model.
            (i.e. binding_prediction.language_model.Elmo)
        path : filepath
            Filepath of the pretrained model.
        """
        self.lm = cls(path, device=device)

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
