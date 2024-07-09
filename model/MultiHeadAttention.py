import torch
import torch.nn as nn

from model.Attention import scaledDotProductAttention


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

        self.Wq = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)
        self.Wk = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)
        self.Wv = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)
        self.Wo = nn.Linear(embeddingDim, embeddingDim, bias=False, dtype=self.dtype)
        self.attention = scaledDotProductAttention(
            contextLength, self.headDim, self.dropout, self.dtype
        )

    def splitHeads(self, x):
        batch_size, context_length, embedding_dim = x.size()
        x = x.reshape(batch_size, context_length, self.numHeads, self.headDim)
        x = x.transpose(1, 2)
        return x

    def combineHeads(self, x):
        batch_size, num_heads, context_length, head_dim = x.size()
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, context_length, self.embeddingDim)
        return x

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = self.splitHeads(q)
        k = self.splitHeads(k)
        v = self.splitHeads(v)

        out = self.attention(q, k, v, self.mask)
        out = self.combineHeads(out)
        out = self.Wo(out)

        return out
