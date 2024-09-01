import torch
import pytorch_lightning as pl
import torch.utils.data as data
from lightning.data import optimize, StreamingDataLoader, StreamingDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer, tensorizer):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.output = config.output
        self.val_frac = config.val_percentage / 100
        self.test_frac = config.test_percentage / 100
        self.random_seed = config.random_seed
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.Tokenizer = tokenizer
        self.Tensorizer = tensorizer

    def preprocess(self):
        tokens = self.Tokenizer.tokenize()
        self.data = self.Tensorizer.tensorize(tokens)

    def setup(self, stage: str = None):
        self.preprocess()
        self.dataset = data.TensorDataset(self.data[0], self.data[1])
        self.val_len = int(self.val_frac * len(self.dataset))
        self.test_len = int(self.test_frac * len(self.dataset))
        self.train_len = len(self.dataset) - self.val_len - self.test_len

        self.train, self.val, self.test = data.random_split(
            self.dataset,
            [
                self.train_len,
                self.val_len,
                self.test_len,
            ],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
