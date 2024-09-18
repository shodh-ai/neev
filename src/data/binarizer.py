from glob import glob
import json
from pathlib import Path
from litdata import optimize, TokensLoader
from functools import partial
import zstandard as zstd
import json
import os
import json
import numpy as np
import torch

class Binarizer:
    def __init__(self,tokenizer,input,output):
        self.tokenizer = tokenizer
        self.input_dir = input
        self.output_dir = output

    def tokenize_fn(self, filepath):
        """Tokenize a file using the provided tokenizer."""
        if filepath.endswith(".zst"):
            with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for row in f:
                    text = json.loads(row)["text"]
                    text_ids = self.tokenizer.encode(text, out_type=int)
                    text_ids = torch.tensor(text_ids, dtype=torch.int)

                    yield text_ids
        elif filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    text = json.loads(row)["text"]
                    text_ids = self.tokenizer.encode(text, out_type=int)
                    text_ids = torch.tensor(text_ids, dtype=torch.int)
                    yield text_ids
        elif filepath.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    text_ids = self.tokenizer.encode(line, out_type=int)
                    text_ids = torch.tensor(text_ids, dtype=torch.int)
                    yield text_ids
        else:
            raise ValueError(f"Unsupported file type for {filepath}")


    def _process_directory(self, dir, subdir):
        input_dir = os.path.join(dir, subdir)
        
        inputs = [str(file) for file in Path(input_dir).rglob("*.zst")] + [str(file) for file in Path(input_dir).rglob("*.jsonl")] + [str(file) for file in Path(input_dir).rglob("*.txt")]
        print("<<<<<<<<>>>>>>>>>>> inputs files are",inputs)
        optimize(
            fn=partial(self.tokenize_fn),
            inputs=inputs,
            output_dir=os.path.join(self.output_dir, subdir),
            chunk_size=(2049 * 8012),
            item_loader=TokensLoader(),
        )


    def make_bin(self):
        """Process the 'train', 'test', and 'val' subdirectories."""
        for subdir in ["train", "test", "val"]:
            self._process_directory(self.input_dir, subdir)
