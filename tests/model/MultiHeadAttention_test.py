import sys
import unittest
import torch

sys.path.append('/home/shodh/framework')

from model.MultiHeadAttention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.batchSize = 2
        self.contextLength = 5
        self.embeddingDim = 8
        self.numHeads = 2
        self.dropout = 0.1

        self.multi_head_attention = MultiHeadAttention(
            self.batchSize, self.contextLength, self.embeddingDim, self.numHeads, self.dropout
        )

        self.x = torch.randn(self.batchSize, self.contextLength, self.embeddingDim, requires_grad=True)

    def test_multi_head_attention_output_shape(self):
        output = self.multi_head_attention(self.x)
        expected_shape = (self.batchSize, self.contextLength, self.embeddingDim)
        self.assertEqual(output.shape, expected_shape)

    def test_multi_head_attention_no_nan(self):
        output = self.multi_head_attention(self.x)
        self.assertFalse(torch.isnan(output).any())

    def test_multi_head_attention_grad(self):
        output = self.multi_head_attention(self.x)
        output.sum().backward()
        self.assertIsNotNone(self.x.grad)

    def test_split_heads_shape(self):
        q = self.multi_head_attention.Wq(self.x)
        split_q = self.multi_head_attention.splitHeads(q)
        expected_shape = (self.batchSize, self.numHeads, self.contextLength, self.embeddingDim // self.numHeads)
        self.assertEqual(split_q.shape, expected_shape)

    def test_combine_heads_shape(self):
        q = self.multi_head_attention.Wq(self.x)
        split_q = self.multi_head_attention.splitHeads(q)
        combined_q = self.multi_head_attention.combineHeads(split_q)
        expected_shape = (self.batchSize, self.contextLength, self.embeddingDim)
        self.assertEqual(combined_q.shape, expected_shape)
