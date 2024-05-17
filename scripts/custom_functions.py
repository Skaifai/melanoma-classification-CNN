import torch
import torch.nn as nn


def softplus(x):
    return torch.log(1 + torch.exp(-x))


class ProposedActivation(nn.Module):
    def __init__(self):
        super(ProposedActivation, self).__init__()

    @staticmethod
    def forward(x):
        beta = x * torch.e ** x
        proposed = beta * torch.tanh(softplus(x))
        return proposed
