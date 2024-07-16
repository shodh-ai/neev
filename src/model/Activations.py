import torch
import torch.nn as nn


class ReLU(nn.ReLU):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return super().forward(x)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def forward(self, x):
        return super().forward(x)


class ELU(nn.ELU):
    def __init__(self):
        super(ELU, self).__init__()

    def forward(self, x):
        return super().forward(x)


class GELU(nn.GELU):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return super().forward(x)


class SiLU(nn.SiLU):
    """Also known as Swish"""

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return super().forward(x)


class GeGLU(nn.Module):
    def __init__(self):
        super(GeGLU, self).__init__()
        self.GELULayer = nn.GELU()

    def forward(self, x):
        return x * self.GELULayer(x)


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.SiLULayer = nn.SiLU()

    def forward(self, x):
        return x * self.SiLULayer(x)


class ReGLU(nn.Module):
    def __init__(self):
        super(ReGLU, self).__init__()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return x * self.ReLU(x)


class LeGLU(nn.Module):
    def __init__(self):
        super(LeGLU, self).__init__()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        return x * self.LeakyReLU(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))
