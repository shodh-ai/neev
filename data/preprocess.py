import torch
import pytorch_lightning as pl
import torch.utils.data as data

from utils import preprocess


class DataModule(pl.LightningDataModule):
    def __init__(self, train_config, preprocess_config):
        super().__init__()
        self.train_config = train_config
        self.preprocess_config = preprocess_config

    def setup(self, stage: str = None):
        self.data = preprocess(
            self.preprocess_config, self.train_config["context_length"]
        )
        self.vocab_size = self.preprocess_config["vocab_size"]
        self.dataset = data.TensorDataset(self.data[0], self.data[1])
        self.val_frac = self.preprocess_config["val_percent"] / 100
        self.test_frac = self.preprocess_config["test_percent"] / 100
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
            generator=torch.Generator().manual_seed(
                self.preprocess_config["random_state"]
            ),
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=4,
        )
