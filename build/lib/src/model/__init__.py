import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics.text
from deepspeed.ops.adam import FusedAdam

from ..model.Decoder import Decoder
from ..model.Normalizations import LayerNorm
from ..model.PositionalEncoding import Learned
from ..model.Loss import ChunkedCrossEntropyLoss
from ..model.Scheduler import CosineAnnealingWarmRestartsDecay


class Transformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.batchSize = config.batch_size
        self.contextLength = config.context_length
        self.embeddingDim = config.embedding_dimension
        self.numHeads = config.num_heads
        self.numLayers = config.num_layers
        self.dropout = config.dropout
        self.vocabSize = config.vocab_size
        self.external_dtype = config.dtype
        self.learningRate = config.learning_rate
        self.weightDecay = config.weight_decay
        self.T_0 = config.T_0
        self.T_mult = config.T_mult
        self.eta_min = config.eta_min
        self.decay = config.lr_decay
        self.config = config

        self.inputEmbed = nn.Embedding(
            self.vocabSize, self.embeddingDim, dtype=self.external_dtype
        )
        self.pe = Learned(self.contextLength, self.embeddingDim, self.external_dtype)
        self.decoder = Decoder(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.numHeads,
            self.numLayers,
            self.dropout,
            self.external_dtype,
            self.config
        )
        self.final_norm = LayerNorm(self.embeddingDim)
        self.linear = nn.Linear(
            self.embeddingDim, self.vocabSize, dtype=self.external_dtype
        )

        self.loss_fn = ChunkedCrossEntropyLoss(ignore_index=0)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.vocabSize
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=self.vocabSize
        )
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=self.vocabSize
        )
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.vocabSize)
        self.ppl = torchmetrics.text.Perplexity()

    def forward(self, x):
        x = self.inputEmbed(x)
        x = self.pe(x)
        x = self.decoder(x)
        x = self.final_norm(x)
        x = self.linear(x)
        x = x.reshape(-1, self.vocabSize)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        accuracy = self.accuracy(output, y)
        # ppl = self.ppl(output, y)
        dict_log = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(dict_log, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        accuracy = self.accuracy(output, y)
        f1_score = self.f1_score(output, y)
        precision = self.precision(output, y)
        recall = self.recall(output, y)
        dict_log = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_f1_score": f1_score,
            "test_precision": precision,
            "test_recall": recall,
        }
        self.log_dict(dict_log, sync_dist=True)
        return loss

    def predict_step(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return output

    def configure_optimizers(self):
        optimizer = FusedAdam(
            self.parameters(),
            lr=self.learningRate,
            weight_decay=self.weightDecay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        lr_scheduler = {
            "scheduler": CosineAnnealingWarmRestartsDecay(
                optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
                decay=self.decay,
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]
