from glob import glob
import json
from pathlib import Path
from litdata import optimize, TokensLoader
from functools import partial
import zstandard as zstd
import json
import os
from .HFTokenizer import Tokenizer as HFTokenizer


from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC


class JsonlTokenizer:
    def __init__(self,input_dir,output_dir,vocab_save_path,tokenizer_type,vocab_size):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.vocab_save_path = vocab_save_path
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        

    def load_jsonl(self, input_path, quiet=True) -> list:
        """
        Read list of objects from a JSON lines file.
        """
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        if not quiet:
            print("Loaded {} records from {}".format(len(data), input_path))
        return data

    def json_iterator(self, text_key="text", EOT_token="<|endoftext|>"):
        
        all_jsonls = glob(f"{self.input_dir}/**/*.jsonl", recursive=True)
        for j in all_jsonls:
            data = self.load_jsonl(j)
            for doc in data:
                yield f"{EOT_token}{doc[text_key]}"

    def train_tokenizer(
        self,
        input_dir: str,
        save_path: str,
        tokenizer_type: str = "BPE",
        vocab_size: int = 32000,
    ):
        """
        Trains a tokenizer on all the json files in `input_dir` and saves it to `save_path`

        :param input_dir: input directory containing jsonl files
        :param save_path: path to save tokenizer to
        :param tokenizer_type: type of tokenizer to train.
        :param vocab_size: int, size of tokenizer's vocab
        :return:
        """

        if tokenizer_type == "BPE":
            model = models.BPE()
        else:
            raise NotImplementedError(
                f"Tokenizer type {tokenizer_type} not implemented"
            )
        tokenizer = Tokenizer(model)

        # Customize pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.normalizer = NFKC()

        # And then train

        # Adding BOS and EOS tokens
        special_tokens = ["<|endoftext|>", "<|padding|>", "<|unknown|>"]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )

        tokenizer.train_from_iterator(
            self.json_iterator(EOT_token=special_tokens[0]),
            trainer,
            length=None,
        )
        

        # And Save it
        if save_path:
            print("<<<<<<<<>>>>>>>>>>>>",save_path)
            tokenizer.save(save_path, pretty=True)
            print(f"Tokenizer saved at {save_path}")
        
        self.tokenizer = HFTokenizer(save_path)
    


    def tokenize_fn(self, filepath):
        """Tokenize a file using the provided tokenizer."""
        if filepath.endswith(".zst"):
            # Handle .jsonl.zst files (compressed)
            with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for row in f:
                    text = json.loads(row)["text"]
                    text_ids = self.tokenizer.encode(text, bos=False, eos=True)
                    yield text_ids
        elif filepath.endswith(".jsonl"):
            # Handle .jsonl files (uncompressed)
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    text = json.loads(row)["text"]
                    text_ids = self.tokenizer.encode(text, bos=False, eos=True)
                    yield text_ids
        else:
            raise ValueError(f"Unsupported file type for {filepath}")

    def _process_directory(self, dir, subdir):
        """Helper function to process a specific subdirectory."""
        input_dir = os.path.join(dir, subdir)
        
        inputs = [str(file) for file in Path(input_dir).rglob("*.zst")] + [str(file) for file in Path(input_dir).rglob("*.jsonl")]
        
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

    def tokenize(self):
        self.train_tokenizer(
            input_dir=self.input_dir,
            save_path=self.vocab_save_path,
            tokenizer_type=self.tokenizer_type,
            vocab_size=self.vocab_size,
        )
        self.make_bin()
