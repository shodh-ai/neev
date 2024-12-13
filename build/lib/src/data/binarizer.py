import os
import torch
from pathlib import Path
from functools import partial
from litdata import optimize, TokensLoader
from tokenizers import Tokenizer


class Binarizer:
    def __init__(self, tokenizer, input, output, context_length, entries, num_workers):
        self.tokenizer_path = tokenizer
        self.input_dir = input
        self.output_dir = output
        self.context_length = context_length
        self.entries = entries
        self.num_workers = num_workers

    def tokenize_fn(self, filepath):
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                text_ids = tokenizer.encode(line).ids
                text_ids = torch.tensor(text_ids, dtype=torch.int)
                yield text_ids

    def _process_directory(self, dir, subdir):
        input_dir = os.path.join(dir, subdir)
        inputs = [str(file) for file in Path(input_dir).rglob("*.txt")]
        optimize(
            fn=partial(self.tokenize_fn),
            inputs=inputs,
            output_dir=os.path.join(self.output_dir, subdir),
            chunk_size=((self.context_length + 1) * self.entries),
            item_loader=TokensLoader(),
            num_workers=self.num_workers,
        )

    def make_bin(self):
        for subdir in ["train", "test", "val"]:
            self._process_directory(self.input_dir, subdir)
