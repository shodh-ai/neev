from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class NeevTokenizer:
    def __init__(self, input, output, vocab_size):
        self.txt_files = [str(file) for file in Path(input).rglob("*.txt")]
        self.output = output
        path = self.output + "/tokenizer"
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=["<|endoftext|>"],
        )
        tokenizer.train(self.txt_files, trainer)
        tokenizer.save(path)
        self.path = path
        self.tokenizer = tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenizer_path(self):
        return self.path
