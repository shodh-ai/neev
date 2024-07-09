import unittest
import torch
import sys

sys.path.append("/home/shodh/framework")

from model.Decoder import Decoder


class TestDecoder(unittest.TestCase):

    def setUp(self):
        self.batchSize = 2
        self.contextLength = 5
        self.embeddingDim = 8
        self.numHeads = 2
        self.numLayers = 3
        self.dropout = 0.1

        self.decoder = Decoder(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.numHeads,
            self.numLayers,
            self.dropout,
        )

        self.x = torch.randn(
            self.batchSize, self.contextLength, self.embeddingDim, requires_grad=True
        )

    def test_decoder_output_shape(self):
        output = self.decoder(self.x)
        expected_shape = (self.batchSize, self.contextLength, self.embeddingDim)
        self.assertEqual(output.shape, expected_shape)

    def test_decoder_no_nan(self):
        output = self.decoder(self.x)
        self.assertFalse(torch.isnan(output).any())

    def test_decoder_grad(self):
        output = self.decoder(self.x)
        output.sum().backward()
        self.assertIsNotNone(self.x.grad)
