import sys
import unittest
import torch
import torch.nn.functional as F

path = "/home/shodh/framework"
sys.path.append(path)

from model.Loss import ChunkedCrossEntropyLoss


class TestChunkedCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = ChunkedCrossEntropyLoss(chunk_size=2)

    def test_no_chunking(self):
        logits = torch.randn(6, 10, requires_grad=True)
        targets = torch.randint(0, 10, (6,))
        loss = self.loss_fn(logits, targets)
        expected_loss = F.cross_entropy(logits, targets)
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_chunking(self):
        logits = torch.randn(6, 10, requires_grad=True)
        targets = torch.randint(0, 10, (6,))
        self.loss_fn.chunk_size = 2
        loss = self.loss_fn(logits, targets)
        logits_chunks = torch.split(logits, 2)
        targets_chunks = torch.split(targets, 2)
        expected_loss = torch.cat(
            [
                F.cross_entropy(l, t, reduction="none")
                for l, t in zip(logits_chunks, targets_chunks)
            ]
        ).mean()
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_ignore_index(self):
        logits = torch.randn(6, 10, requires_grad=True)
        targets = torch.tensor([1, 0, -100, 4, -100, 5])
        self.loss_fn.ignore_index = -100
        loss = self.loss_fn(logits, targets)
        expected_loss = F.cross_entropy(logits, targets, ignore_index=-100)
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_list_input_no_chunking(self):
        logits = [torch.randn(2, 10, requires_grad=True) for _ in range(3)]
        targets = torch.randint(0, 10, (6,))  # Change targets shape to (6,)
        self.loss_fn.chunk_size = 0

        # Compute loss using the loss function
        loss = self.loss_fn(logits, targets)

        # Combine logits into a single tensor with the correct batch size
        logits_combined = torch.cat(logits, dim=0)

        # Expected loss calculation
        expected_loss = F.cross_entropy(
            logits_combined, targets, ignore_index=self.loss_fn.ignore_index
        )

        # Verify that the calculated loss matches the expected loss
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_list_input_with_chunking(self):
        logits = [torch.randn(2, 10, requires_grad=True) for _ in range(3)]
        targets = torch.randint(0, 10, (2, 1))
        self.loss_fn.chunk_size = 2
        loss = self.loss_fn(logits, targets)
        logits_combined = [l.reshape(-1, 10) for l in logits]
        targets_combined = [t.reshape(-1) for t in torch.split(targets, 1, dim=1)]
        expected_loss = torch.cat(
            [
                F.cross_entropy(l, t, reduction="none")
                for l, t in zip(logits_combined, targets_combined)
            ]
        ).mean()
        self.assertTrue(torch.allclose(loss, expected_loss))
