import torch.nn as nn
from model.Activations import SwiGLU


class FeedForward(nn.Module):

    def __init__(self, batchSize, contextLength, embeddingDim, dropout, dtype):
        super(FeedForward, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout
        self.dtype = dtype

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.inTranform = nn.Linear(embeddingDim, embeddingDim, dtype=self.dtype)
        self.activation = SwiGLU()
        self.outTransform = nn.Linear(embeddingDim, embeddingDim, dtype=self.dtype)

    def forward(self, x):
        x = self.inTranform(x)
        x = self.dropoutLayer(x)
        x = self.activation(x)
        x = self.dropoutLayer(x)
        x = self.outTransform(x)
        return x
