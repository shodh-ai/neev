import torch
import os
import pytorch_lightning as pl
from lightning.data import optimize, StreamingDataLoader, StreamingDataset

import pytorch_lightning as pl
from lightning.data import (
    StreamingDataset,
    StreamingDataLoader,
)
from litdata import TokensLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, input_dir,batch_size, vocab_size,context_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.input_dir = input_dir
        self.batch_size = batch_size


    def setup(self, stage: str = None):
        
        self.vocab_size = self.vocab_size
        input_dir=os.path.join(self.input_dir, "train")
        print("<<<<>>>> loading data from ",input_dir)
        self.train = StreamingDataset(
            input_dir=os.path.join(self.input_dir, "train"),
            item_loader=TokensLoader(
                block_size=self.context_length + 1
            ),
            shuffle=False,
        )
        input_dir=os.path.join(self.input_dir, "val")
        print("<<<<>>>> loading data from ",input_dir)
        self.val = StreamingDataset(
            input_dir=os.path.join(self.input_dir, "val"),
            item_loader=TokensLoader(
                block_size=self.context_length + 1
            ),
            shuffle=False,
        )
        input_dir=os.path.join(self.input_dir, "test")
        print("<<<<>>>> loading data from ",input_dir)
        self.test = StreamingDataset(
            input_dir=os.path.join(self.input_dir, "test"),
            item_loader=TokensLoader(
                block_size=self.context_length + 1
            ),
            shuffle=False,
        )

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
           
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            
        )
