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
        self.lr_decay = config.lr_decay
        self.decoder_architechture = config.decoder_architechture

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
            self.decoder_architechture
        )
        self.final_norm = LayerNorm(self.embeddingDim)
        self.linear = nn.Linear(
            self.embeddingDim, self.vocabSize, dtype=self.external_dtype
        ) 

        self.loss_fn = ChunkedCrossEntropyLoss(ignore_index=0)
        self.ppl = torchmetrics.text.Perplexity()

    def forward(self, x):
        x = self.inputEmbed(x)
        x = self.pe(x)
        x = self.decoder(x)
        x = self.final_norm(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch[:, : self.contextLength]
        y = batch[:, 1:].long()

        output = self.forward(x)
        loss = self.loss_fn(
            output.reshape(output.shape[0] * output.shape[1], self.vocabSize), y
        )

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, : self.contextLength]
        y = batch[:, 1:].long()

        output = self.forward(x)
        loss = self.loss_fn(
            output.reshape(output.shape[0] * output.shape[1], self.vocabSize), y
        )
        val_ppl = self.ppl(output, y)
        dict_log = {
            "val_loss": loss,
            "val_ppl": val_ppl,
        }
        self.log_dict(dict_log, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[:, : self.contextLength]
        y = batch[:, 1:].long()
        output = self.forward(x)
        loss = self.loss_fn(
            output.reshape(output.shape[0] * output.shape[1], self.vocabSize), y
        )
        test_ppl = self.ppl(output, y)
        dict_log = {
            "test_loss": loss,
            "test_ppl": test_ppl,
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
                decay=self.lr_decay,
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]
