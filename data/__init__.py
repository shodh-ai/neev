import os
import json
import torch

from utils.vocab import vocab_builder
from utils.tokenization import tokenize
from utils.dataset_creation import dataset_creation


def preprocess(config, seq_len):
    data_dir = config["data_dir"]
    if os.path.exists(f"{data_dir}/data.pt"):
        data = torch.load(f"{data_dir}/data.pt")
    else:
        tokens = tokenize(config)
        vocab_builder(config)
        data = dataset_creation(tokens, config, seq_len)
    return data
