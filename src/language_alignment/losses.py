import torch
import torch.nn as nn
import torch.nn.functional as F


class CCAloss(object):
    """
    This is from the DeepCCA repository
    Author: Michaelvll
    """
    def __init__(self, outdim_size, use_all_singular_values=True):
        super(CCAloss, self).__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values

    def __call__(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper.
        There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        assert torch.isnan(H1).sum().item() == 0
        assert torch.isnan(H2).sum().item() == 0
        H1, H2 = torch.squeeze(H1).t(), torch.squeeze(H2).t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        #         print(H1.size())

        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0
        eye1 = torch.eye(o1).to(H1bar.device)
        eye2 = torch.eye(o2).to(H2bar.device)
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(
            H1bar, H1bar.t()) + r1 * eye1
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(
            H2bar, H2bar.t()) + r2 * eye2
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
        #         print(Tval.size())

        # all singular values are used to calculate the correlation
        tmp = torch.trace(torch.matmul(Tval.t(), Tval))
        # print(tmp)
        corr = torch.sqrt(tmp)
        # assert torch.isnan(corr).item() == 0

        return -corr


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        #self.loss = nn.MarginRankingLoss()

    """ From FaceNet
    https://arxiv.org/pdf/1503.03832.pdf

    Deep Metric Learing Using Triplet Network
    https://arxiv.org/pdf/1412.6622.pdf
    """
    def __call__(self, xy, xz, alpha=0.1):
        xy2 = torch.pow(xy, 2)
        xz2 = torch.pow(xz, 2)
        loss = torch.clamp(xy2 - xz2 + alpha, min=0)
        return loss

class RankingLoss(nn.Module):
    """ From Bayesian Personalized Ranking
    https://arxiv.org/pdf/1205.2618.pdf
    """
    def __call__(self, xy, xz):
        diff = xy - xz
        losses = F.logsigmoid(diff)
        #losses = sum(score)
        #losses = diff ** 2
        return -1 * losses
