import torch
import torch.nn as nn

decorator = torch.compile

@decorator
def rms_norm(x, eps: float):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)

@decorator
def crms_norm(x, eps: float):
    discarded_element = x.sum(dim=-1, keepdim=True)
    return x * torch.rsqrt(
        (x.square().sum(dim=-1, keepdim=True) + discarded_element.square())
        / (x.shape[-1] + 1)
        + eps
    )


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
        return rms_norm(x, self.eps)


class cRMSNorm(nn.Module):
    def __init__(self, embeddingDim, eps=1e-8):
        super(cRMSNorm, self).__init__()

        self.embeddingDim = embeddingDim
        self.eps = eps

    def forward(self, x):
        return crms_norm(x, self.eps)


class LinearZeroMeanOutput(nn.Linear):
    def __init__(self, embeddingDim):
        super(cRMSNorm, self).__init__()
        self.embeddingDim = embeddingDim

    def forward(self, x):
        zero_mean_weight = self.weight - self.weight.mean(dim=0, keepdim=True)
        zero_mean_bias = self.bias - self.bias.mean()
        return nn.functional.linear(x, zero_mean_weight, zero_mean_bias)
