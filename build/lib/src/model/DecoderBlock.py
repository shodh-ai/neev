import torch.nn as nn

from ..model.FeedForward import FeedForward
from ..model.MultiHeadAttention import MultiHeadAttention
from ..model.Normalizations import LayerNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        batchSize,
        contextLength,
        embeddingDim,
        numHeads,
        dropout,
        dtype
    ):
        super(DecoderBlock, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.dropout = dropout
        self.dtype = dtype

        self.MHA = MultiHeadAttention(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.numHeads,
            self.dropout,
            self.dtype,
        )
        self.FF = FeedForward(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.dropout,
            self.dtype,
        )
        self.normalisation_mha = LayerNorm(self.embeddingDim)
        self.normalisation_ffn = LayerNorm(self.embeddingDim)

    def forward(self, x):
        h = self.FF(self.normalisation_ffn(x))
        x = x + self.MHA(self.normalisation_mha(x)) + h
        return x
