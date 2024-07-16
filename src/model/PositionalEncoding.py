import math
import torch
import torch.nn as nn


class Learned(nn.Module):
    def __init__(self, contextLength, embeddingDim, dtype):
        super(Learned, self).__init__()

        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dtype = dtype

        self.pos_embedding_layer = nn.Embedding(
            self.contextLength, self.embeddingDim, dtype=self.dtype
        )

    def forward(self, x):
        pos_indices = torch.arange(self.contextLength, device=x.device)
        pos_embedding = self.pos_embedding_layer(pos_indices).to(self.dtype)
        x = x + pos_embedding
        return x


class Cosine(nn.Module):
    def __init__(self, contextLength, embeddingDim, dtype):
        super(Cosine, self).__init__()

        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dtype = dtype

        self.posVector = torch.zeros(
            self.contextLength, self.embeddingDim, dtype=self.dtype
        )
        for pos in range(self.contextLength):
            for i in range(0, self.embeddingDim, 2):
                self.posVector[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i) / self.embeddingDim))
                )
                self.posVector[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * i) / self.embeddingDim))
                )
        self.posVector = self.posVector.to(self.dtype)

    def forward(self, x):
        x = x + self.posVector.to(x.device)
        return x


class RoPE(nn.Module):
    def __init__(self, context_length, embedding_dim, dtype):
        super(RoPE, self).__init__()
        self.dim = embedding_dim
        self.context_length = context_length
        self.dtype = dtype
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embedding_dim, 2) / embedding_dim)
        )
        self.inv_freq = self.inv_freq.to(dtype)

    def forward(self, x):
        seq_len = self.context_length
        t = torch.arange(seq_len, dtype=self.dtype, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
        pos_enc = (
            torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
            .to(x.device)
            .to(self.dtype)
        )
        return x * pos_enc
