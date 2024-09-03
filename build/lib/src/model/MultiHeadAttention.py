import torch
import torch.nn as nn

from ..model.Attention import scaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self, batchSize, contextLength, embeddingDim, numHeads, dropout, dtype
    ):
        super(MultiHeadAttention, self).__init__()

        assert embeddingDim % numHeads == 0
        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.dropout = dropout
        self.dtype = dtype

        self.headDim = embeddingDim // numHeads
        self.mask = torch.triu(torch.ones(contextLength, contextLength), diagonal=1)
        self.mask = self.mask.masked_fill(self.mask == 1, float(-1e9))
        self.mask = self.mask.to(self.dtype)

        self.Wqkv = nn.Linear(embeddingDim, 3 * embeddingDim, dtype=self.dtype)
        self.Wo = nn.Linear(embeddingDim, embeddingDim, dtype=self.dtype)
        self.attention = scaledDotProductAttention(
            contextLength, self.headDim, self.dropout, self.dtype
        )

    def splitHeads(self, x):
        x = x.reshape(-1, self.contextLength, self.numHeads, self.headDim)
        x = x.transpose(1, 2)
        return x

    def combineHeads(self, x):
        x = x.transpose(0, 1)
        x = x.reshape(self.contextLength, -1, self.embeddingDim)
        return x

    def forward(self, x):
        qkv = self.Wqkv(x)
        qkv = qkv.reshape(self.contextLength, -1, self.numHeads, 3 * self.headDim)
        q, k, v = qkv.split(self.headDim, -1)
        q = self.splitHeads(q)
        k = self.splitHeads(k)
        v = self.splitHeads(v)

        q= q.permute(2,0,1,3)
        k= k.permute(2,0,1,3)
        v= v.permute(2,0,1,3)

        out = self.attention(q, k, v, self.mask)
        out = self.combineHeads(out)
        out = self.Wo(out)
        out = out.transpose(0, 1)

        return out
