import os
import time
import json
import string
from sentencepiece import SentencePieceProcessor
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

punctuations = string.punctuation


def measure_time(start_time=None):
    if start_time is None:
        return time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def tokenizer(config):
    sp = SentencePieceProcessor(
        os.path.join(config["vocab_dir"], config["tokenizer"]) + ".model"
    )
    return sp


def vocab(config):
    vocab = json.load(open(f"{config['vocab_dir']}/vocab.json", "r"))
    return vocab


def rev_vocab(config):
    rev_vocab = json.load(open(f"{config['vocab_dir']}/rev_vocab.json", "r"))
    return rev_vocab


def load_checkpoint(checkpoint_version):
    checkpoint_dir = "./logs/checkpoints"
    dirs = [
        d
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if len(dirs) > 1:
        dirs.remove("best-checkpoint.ckpt")
        best_checkpoint = max(dirs)
    else:
        best_checkpoint = dirs[0]
    if checkpoint_version:
        best_checkpoint = checkpoint_version

    input_path = os.path.join(checkpoint_dir, best_checkpoint)
    output_path = os.path.join(checkpoint_dir, "best-checkpoint-fp32.ckpt")
    convert_zero_checkpoint_to_fp32_state_dict(input_path, output_path)
    return output_path
