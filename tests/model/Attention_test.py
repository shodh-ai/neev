import unittest
import torch
import sys

sys.path.append('/home/shodh/framework')

from model.Attention import scaledDotProductAttention, additiveAttention

class TestAttention(unittest.TestCase):

    def setUp(self):
        self.contextLength = 5
        self.embeddingDim = 8
        self.dropout = 0.1
        self.batch_size = 2
        self.seq_len = 5

        self.q = torch.randn(self.batch_size, self.seq_len, self.embeddingDim, requires_grad=True)
        self.k = torch.randn(self.batch_size, self.seq_len, self.embeddingDim, requires_grad=True)
        self.v = torch.randn(self.batch_size, self.seq_len, self.embeddingDim, requires_grad=True)
        self.mask = torch.zeros(self.batch_size, self.seq_len, self.seq_len)
        
        self.scaled_dot_product_attention = scaledDotProductAttention(self.contextLength, self.embeddingDim, self.dropout)
        self.additive_attention = additiveAttention(self.contextLength, self.embeddingDim, self.dropout)

    def test_scaled_dot_product_attention_output_shape(self):
        output = self.scaled_dot_product_attention(self.q, self.k, self.v, self.mask)
        expected_shape = (self.batch_size, self.seq_len, self.embeddingDim)
        self.assertEqual(output.shape, expected_shape)

    def test_additive_attention_output_shape(self):
        output = self.additive_attention(self.q, self.k, self.v, self.mask)
        expected_shape = (self.batch_size, self.seq_len, self.embeddingDim)
        self.assertEqual(output.shape, expected_shape)

    def test_scaled_dot_product_attention_no_nan(self):
        output = self.scaled_dot_product_attention(self.q, self.k, self.v, self.mask)
        self.assertFalse(torch.isnan(output).any())

    def test_additive_attention_no_nan(self):
        output = self.additive_attention(self.q, self.k, self.v, self.mask)
        self.assertFalse(torch.isnan(output).any())

    def test_scaled_dot_product_attention_grad(self):
        output = self.scaled_dot_product_attention(self.q, self.k, self.v, self.mask)
        output.sum().backward()
        self.assertIsNotNone(self.q.grad)
        self.assertIsNotNone(self.k.grad)
        self.assertIsNotNone(self.v.grad)

    def test_additive_attention_grad(self):
        output = self.additive_attention(self.q, self.k, self.v, self.mask)
        output.sum().backward()
        self.assertIsNotNone(self.q.grad)
        self.assertIsNotNone(self.k.grad)
        self.assertIsNotNone(self.v.grad)

if __name__ == '__main__':
    unittest.main()
