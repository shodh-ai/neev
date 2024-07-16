import math
import torch
import torch.nn as nn


class scaledDotProductAttention(nn.Module):
    def __init__(self, contextLength, embeddingDim, dropout, dtype):
        super(scaledDotProductAttention, self).__init__()
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout
        self.dtype = dtype
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        k = k.transpose(-2, -1)
        scores = torch.matmul(q, k)
        scores = scores / math.sqrt(self.embeddingDim)
        scores = scores + mask.to(self.dtype).to(scores.device)
        attention = nn.Softmax(dim=-1)(scores).to(self.dtype)
        out = torch.matmul(attention, v)
        out = self.dropoutLayer(out)
        return out


class additiveAttention(nn.Module):
    def __init__(self, contextLength, embeddingDim, dropout, dtype):
        super(additiveAttention, self).__init__()
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout
        self.dtype = dtype

        self.queryExpand = nn.Linear(embeddingDim, contextLength, dtype=self.dtype)
        self.keyExpand = nn.Linear(embeddingDim, contextLength, dtype=self.dtype)
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        q = self.queryExpand(q)
        k = self.keyExpand(k)
        energy = torch.tanh(q + k)
        energy = energy + mask.to(self.dtype).to(energy.device)
        attention = nn.Softmax(dim=-1)(energy).to(self.dtype)
        out = torch.matmul(attention, v)
        out = self.dropoutLayer(out)
        return out
