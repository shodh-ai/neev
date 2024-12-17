import pytorch_lightning as pl
from lightning.data import StreamingDataset, StreamingDataLoader
from litdata import TokensLoader
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_path = config.train_bin_path
        self.val_path = config.val_bin_path
        self.test_path = config.test_bin_path
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size
        self.context_length = config.context_length

    def setup(self, stage: str = None):
        self.train = StreamingDataset(
            input_dir=self.train_path,
            item_loader=TokensLoader(block_size=self.context_length + 1),
        )
        self.val = StreamingDataset(
            input_dir=self.val_path,
            item_loader=TokensLoader(block_size=self.context_length + 1),
        )
        self.test = StreamingDataset(
            input_dir=self.test_path,
            item_loader=TokensLoader(block_size=self.context_length + 1),
        )

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
