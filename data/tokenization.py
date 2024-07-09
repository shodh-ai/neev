import os
import glob
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


def tokenize(config):
    txt_files = glob.glob(f"{config['data_dir']}/*.txt")
    input_files = ",".join(txt_files)
    path = os.path.join(config["vocab_dir"], config["tokenizer"])

    SentencePieceTrainer.train(
        input=input_files,
        model_prefix=path,
        vocab_size=config["vocab_size"],
        character_coverage=config["spm_convergence"],
        model_type=config["tokenizer"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
    )

    sp = SentencePieceProcessor(model_file=path + ".model")
    text = "".join([open(file, "r", encoding="utf-8").read() for file in txt_files])
    tokens = sp.encode(text, out_type=int)
    return tokens
