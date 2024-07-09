import unittest
import torch
import sys

sys.path.append('/home/shodh/framework')

from model.DecoderBlock import DecoderBlock

class TestDecoderBlock(unittest.TestCase):

    def setUp(self):
        self.batchSize = 2
        self.contextLength = 5
        self.embeddingDim = 8
        self.numHeads = 2
        self.dropout = 0.1

        self.decoder_block = DecoderBlock(
            self.batchSize, self.contextLength, self.embeddingDim, self.numHeads, self.dropout
        )

        self.x = torch.randn(self.batchSize, self.contextLength, self.embeddingDim, requires_grad=True)

    def test_decoder_block_output_shape(self):
        output = self.decoder_block(self.x)
        expected_shape = (self.batchSize, self.contextLength, self.embeddingDim)
        self.assertEqual(output.shape, expected_shape)

    def test_decoder_block_no_nan(self):
        output = self.decoder_block(self.x)
        self.assertFalse(torch.isnan(output).any())

    def test_decoder_block_grad(self):
        output = self.decoder_block(self.x)
        output.sum().backward()
        self.assertIsNotNone(self.x.grad)


