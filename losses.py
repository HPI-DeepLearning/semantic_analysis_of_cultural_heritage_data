import torch
from torch.autograd import Variable
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, sigma=.1):
        self.sigma = sigma
        super(MMDLoss, self).__init__()

    def forward(self, x, y):
        alpha = 1. / (2 * self.sigma ** 2)

        B = x.size()[0]
        assert (y.size()[0] == B)
        x = x.view(B, -1)
        y = y.view(B, -1)

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K = torch.exp(- alpha * (rx.t() + rx - 2 * xx))
        L = torch.exp(- alpha * (ry.t() + ry - 2 * yy))
        P = torch.exp(- alpha * (rx.t() + ry - 2 * zz))

        beta = (1. / (B * (B - 1)))
        gamma = (2. / (B * B))

        return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.0, measure=False, reduction='sum', max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.reduction = reduction

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.reduction == 'sum':
            return cost_s.sum() + cost_im.sum()
        elif self.reduction == 'mean':
            return cost_s.mean() + cost_im.mean()
        else:
            return cost_s + cost_im
