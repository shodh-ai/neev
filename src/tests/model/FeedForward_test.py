import unittest
import torch
import sys

sys.path.append('/home/shodh/framework')

from model.FeedForward import FeedForward

class TestFeedForward(unittest.TestCase):

    def setUp(self):
        self.batchSize = 2
        self.contextLength = 5
        self.embeddingDim = 8
        self.dropout = 0.1

        self.feed_forward = FeedForward(
            self.batchSize, self.contextLength, self.embeddingDim, self.dropout
        )

        self.x = torch.randn(self.batchSize, self.contextLength, self.embeddingDim, requires_grad=True)

    def test_feed_forward_output_shape(self):
        output = self.feed_forward(self.x)
        expected_shape = (self.batchSize, self.contextLength, self.embeddingDim)
        self.assertEqual(output.shape, expected_shape)

    def test_feed_forward_no_nan(self):
        output = self.feed_forward(self.x)
        self.assertFalse(torch.isnan(output).any())

    def test_feed_forward_grad(self):
        output = self.feed_forward(self.x)
        output.sum().backward()
        self.assertIsNotNone(self.x.grad)

if __name__ == '__main__':
    unittest.main()
