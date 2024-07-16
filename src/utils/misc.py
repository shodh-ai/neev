import os
import time
import json
from sentencepiece import SentencePieceProcessor
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


def measure_time(start_time=None):
    if start_time is None:
        return time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def tokenizer(path):
    sp = SentencePieceProcessor(path + ".model")
    return sp


def vocab(path):
    vocab = json.load(open(path, "r"))
    return vocab


def rev_vocab(path):
    rev_vocab = json.load(open(path, "r"))
    return rev_vocab


def load_checkpoint(checkpoint_path):
    input_path = checkpoint_path
    output_path = os.path.join(
        os.path.dirname(input_path), "fp32_" + os.path.basename(input_path)
    )
    convert_zero_checkpoint_to_fp32_state_dict(input_path, output_path)
    return output_path
