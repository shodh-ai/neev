import os
import torch
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class NeevTokenizer:
    def __init__(self, input, output, vocab_size, context_length):
        self.txt_files = [str(file) for file in Path(input).rglob("*.txt")]
        self.output = output
        self.context_length = context_length
        path = self.output + "/tokenizer"
        if not os.path.exists(path):
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                show_progress=True,
                special_tokens=["<EOS>", "<BOS>", "<PAD>", "<UNK>"],
            )
            tokenizer.train(self.txt_files, trainer)
            tokenizer.save(path)
        self.path = path
        self.tokenizer = Tokenizer.from_file(path)
        self.eos = self.tokenizer.encode("<EOS>").ids[0]
        self.bos = self.tokenizer.encode("<BOS>").ids[0]
        self.pad = self.tokenizer.encode("<PAD>").ids[0]

    def _append_and_shift(self, tensor, value):
        new_tensor = torch.cat([tensor[1:], torch.tensor([value], dtype=tensor.dtype)])
        return new_tensor

    def encode(self, text):
        encoded_text = self.tokenizer.encode(text).ids
        if encoded_text == []:
            return []
        encoded_text = [self.bos] + encoded_text + [self.eos]
        return encoded_text
