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
        # (seqlen,batch,numHeads,headDim) -> (batch,numHeads,seqlen,headDim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # (batch,numHeads,seqlen,headDim) -> (batch*numHeads,seqlen,headDim)
        q = q.reshape(q.size(0) * q.size(1), q.size(2), q.size(3))
        k = k.reshape(k.size(0) * k.size(1), k.size(2), k.size(3))
        v = v.reshape(v.size(0) * v.size(1), v.size(2), v.size(3))

        # (batch*numHeads,seqlen,headDim) -> (batch*numHeads,headDim,seqlen)
        k = torch.transpose(k, 1, 2)

        # (batch*numHeads,seqlen,headDim) * (batch*numHeads,headDim,seqlen) -> (batch*numHeads,seqlen,seqlen)
        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(self.embeddingDim))
        
        scores = scores + mask.to(self.dtype).to(scores.device)
        attention = nn.Softmax(dim=-1)(scores).to(self.dtype)

        # (batch*numHeads,seqlen,seqlen) * (batch*numHeads,seqlen,headDim) =  (batch*numHeads,seqlen,headDim)
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
