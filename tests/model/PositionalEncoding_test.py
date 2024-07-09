import sys
import torch
import unittest
import torch.nn.functional as F
from unittest import TestCase

path = '/home/shodh/framework'
sys.path.append(path)

from model.PositionalEncoding import Learned, RoPE, Cosine

class TestPositionalEncodings(TestCase):

    def setUp(self):
        self.context_length = 10
        self.embedding_dim = 16
        self.batch_size = 2

        self.x = torch.randn(self.batch_size, self.context_length, self.embedding_dim)

    def test_learned_positional_encoding(self):
        model = Learned(self.context_length, self.embedding_dim)
        output = model(self.x)

        self.assertEqual(output.shape, self.x.shape)
        self.assertFalse(torch.equal(output, self.x))

    def test_cosine_positional_encoding(self):
        model = Cosine(self.context_length, self.embedding_dim)
        output = model(self.x)

        self.assertEqual(output.shape, self.x.shape)
        self.assertFalse(torch.equal(output, self.x))

    def test_rope_positional_encoding(self):
        model = RoPE(self.context_length, self.embedding_dim)
        output = model(self.x)

        self.assertEqual(output.shape, self.x.shape)
        self.assertFalse(torch.equal(output, self.x))

    def test_learned_embedding_gradient(self):
        model = Learned(self.context_length, self.embedding_dim)
        output = model(self.x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(model.pos_embedding_layer.weight.grad)
        self.assertTrue((model.pos_embedding_layer.weight.grad != 0).any())

    def test_cosine_no_trainable_parameters(self):
        model = Cosine(self.context_length, self.embedding_dim)
        params = list(model.parameters())
        self.assertEqual(len(params), 0)

    def test_rope_no_trainable_parameters(self):
        model = RoPE(self.context_length, self.embedding_dim)
        params = list(model.parameters())
        self.assertEqual(len(params), 0)

