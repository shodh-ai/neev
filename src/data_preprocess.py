import os
import json
from pathlib import Path
import zstandard as zstd
from .data.tokenizer import NeevTokenizer
from .data.binarizer import Binarizer
from .utils import ConfigReader


class DataPreprocess:
    def __init__(self, config):
        config = ConfigReader(config).read()
        self.input_dir = config.input
        self._convert_to_text()
        self.tokenizer = NeevTokenizer(
            config.input,
            config.output,
            config.vocab_size,
            config.context_length,
        )
        self.binarizer = Binarizer(
            self.tokenizer,
            config.input,
            config.output,
            config.context_length,
            config.entries,
            config.num_workers,
        )

    def _zst_to_txt(self, filepath):
        base_filename = os.path.splitext(filepath)[0]
        txt_file = base_filename + ".txt"

        with open(filepath, "rb") as f_in:
            dctx = zstd.ZstdDecompressor()
            with open(txt_file, "w", encoding="utf-8") as f_out:
                for chunk in dctx.stream_reader(f_in):
                    f_out.write(chunk.decode("utf-8"))

    def _json_to_txt(self, filepath):
        base_filename = os.path.splitext(filepath)[0]
        txt_file = base_filename + ".txt"

        with open(filepath, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)

        with open(txt_file, "w", encoding="utf-8") as f_out:
            for entry in data:
                f_out.write(entry.get("text", "") + "\n")

    def _convert_to_text(self):
        input_file_list = (
            [str(file) for file in Path(self.input_dir).rglob("*.zst")]
            + [str(file) for file in Path(self.input_dir).rglob("*.jsonl")]
            + [str(file) for file in Path(self.input_dir).rglob("*.txt")]
        )
        for file in input_file_list:
            if file.endswith(".zst"):
                self._zst_to_txt(file)
            elif file.endswith(".json") or file.endswith(".jsonl"):
                self._json_to_txt(file)

    def process(self):
        self.binarizer.make_bin()
