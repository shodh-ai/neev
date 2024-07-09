import torch.nn as nn

from model.DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        batchSize,
        contextLength,
        embeddingDim,
        numHeads,
        numLayers,
        dropout,
        dtype,
    ):
        super(Decoder, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.dropout = dropout
        self.dtype = dtype

        self.decoderBlocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.batchSize,
                    self.contextLength,
                    self.embeddingDim,
                    self.numHeads,
                    self.dropout,
                    self.dtype,
                )
                for _ in range(self.numLayers)
            ]
        )

    def forward(self, x):
        for block in self.decoderBlocks:
            x = block(x)
        return x
