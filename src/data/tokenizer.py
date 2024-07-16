import glob
import json
from sentencepiece import SentencePieceTrainer
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, input, output, vocab_size, spm_convergence, tokenizer):
        self.txt_files = glob.glob(f"{input}/*.txt")
        input_files = ",".join(self.txt_files)
        path = output + "/tokenizer"
        self.output = output
        SentencePieceTrainer.train(
            input=input_files,
            model_prefix=path,
            vocab_size=vocab_size,
            character_coverage=spm_convergence,
            model_type=tokenizer,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        )
        self.sp = SentencePieceProcessor(model_file=path + ".model")

    def tokenize(self):
        text = "".join(
            [open(file, "r", encoding="utf-8").read() for file in self.txt_files]
        )
        tokens = self.sp.encode(text, out_type=int)
        self.build_vocab()
        return tokens

    def get_tokenizer(self):
        return self.sp

    def build_vocab(self):
        vocab = {}
        rev_vocab = {}
        vocab = {self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())}
        rev_vocab = {
            id: self.sp.id_to_piece(id) for id in range(self.sp.get_piece_size())
        }
        json.dump(vocab, open(f"{self.output}/vocab.json", "w"))
        json.dump(rev_vocab, open(f"{self.output}/rev_vocab.json", "w"))
