import torch
import unittest
import sys
import math
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("/home/shodh/framework")
from model import Transformer


class TestTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.config = {
            "batch_size": 2,
            "context_length": 5,
            "embedding_dimension": 8,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.1,
            "lr": 0.001,
            "weight_decay": 0.0001,
        }
        self.vocabSize = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer = Transformer(self.config, self.vocabSize).to(self.device)

        x_train = [
            torch.randint(0, self.vocabSize, (self.config["context_length"],))
            for _ in range(5)
        ]
        y_train = [torch.randint(0, self.vocabSize, ()) for _ in range(5)]

        self.dataset = TensorDataset(torch.stack(x_train), torch.stack(y_train))
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.config["batch_size"], num_workers=2
        )

        self.logger = TensorBoardLogger("logs/", name="transformer")
        self.trainer = Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=1,
            min_epochs=1,
            precision="bf16-true",
            log_every_n_steps=1,
            logger=self.logger,
            enable_progress_bar=False,
        )

    def test_transformer_initialization(self):
        self.assertEqual(self.transformer.batchSize, self.config["batch_size"])
        self.assertEqual(self.transformer.contextLength, self.config["context_length"])
        self.assertEqual(
            self.transformer.embeddingDim, self.config["embedding_dimension"]
        )
        self.assertEqual(self.transformer.numHeads, self.config["num_heads"])
        self.assertEqual(self.transformer.numLayers, self.config["num_layers"])
        self.assertEqual(self.transformer.dropout, self.config["dropout"])
        self.assertEqual(self.transformer.vocabSize, self.vocabSize)

    def test_forward(self):
        x, _ = next(iter(self.dataloader))
        x = x.to(self.device)
        output = self.transformer.forward(x)
        expected_shape = (
            self.config["batch_size"],
            self.config["context_length"],
            self.vocabSize,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_training_step(self):
        self.trainer.fit(self.transformer, self.dataloader, self.dataloader)
        self.assertTrue("loss" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["loss"])
        self.assertTrue(math.isnan(self.trainer.logged_metrics["loss"]) == False)

    def test_validation_step(self):
        self.trainer.validate(self.transformer, self.dataloader)
        self.assertTrue("val_loss" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["val_loss"])
        self.assertTrue(math.isnan(self.trainer.logged_metrics["val_loss"]) == False)
        self.assertTrue("val_accuracy" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["val_accuracy"])
        self.assertTrue(
            math.isnan(self.trainer.logged_metrics["val_accuracy"]) == False
        )

    def test_test_step(self):
        self.trainer.test(self.transformer, self.dataloader)
        self.assertTrue("test_loss" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["test_loss"])
        self.assertTrue(math.isnan(self.trainer.logged_metrics["test_loss"]) == False)
        self.assertTrue("test_accuracy" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["test_accuracy"])
        self.assertTrue(
            math.isnan(self.trainer.logged_metrics["test_accuracy"]) == False
        )
        self.assertTrue("test_f1_score" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["test_f1_score"])
        self.assertTrue(
            math.isnan(self.trainer.logged_metrics["test_f1_score"]) == False
        )
        self.assertTrue("test_precision" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["test_precision"])
        self.assertTrue(
            math.isnan(self.trainer.logged_metrics["test_precision"]) == False
        )
        self.assertTrue("test_recall" in self.trainer.logged_metrics)
        self.assertIsNotNone(self.trainer.logged_metrics["test_recall"])
        self.assertTrue(math.isnan(self.trainer.logged_metrics["test_recall"]) == False)


if __name__ == "__main__":
    unittest.main()
