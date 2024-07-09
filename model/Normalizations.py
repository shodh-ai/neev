import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def __init__(self, embeddingDim):
        super(LayerNorm, self).__init__(embeddingDim)
        self.embeddingDim = embeddingDim

    def forward(self, x):
        return super(LayerNorm, self).forward(x)


class RMSNorm(nn.Module):
    def __init__(self, embeddingDim, eps=1e-8):
        super(RMSNorm, self).__init__()

        self.embeddingDim = embeddingDim
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class cRMSNorm(nn.Module):
    def __init__(self, embeddingDim, eps=1e-8):
        super(cRMSNorm, self).__init__()

        self.embeddingDim = embeddingDim
        self.eps = eps

    def forward(self, x):
        discarded_element = x.sum(dim=-1, keepdim=True)
        return x * torch.rsqrt(
            (x.square().sum(dim=-1, keepdim=True) + discarded_element.square())
            / (x.shape[-1] + 1)
            + self.eps
        )
