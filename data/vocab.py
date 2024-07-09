import json
from utils.misc import tokenizer


def vocab_builder(config):
    vocab = {}
    rev_vocab = {}
    tokenizer_instance = tokenizer(config)
    vocab = {
        tokenizer_instance.id_to_piece(id): id
        for id in range(tokenizer_instance.get_piece_size())
    }
    rev_vocab = {
        id: tokenizer_instance.id_to_piece(id)
        for id in range(tokenizer_instance.get_piece_size())
    }
    vocab_dir = config["vocab_dir"]
    json.dump(vocab, open(f"{vocab_dir}/vocab.json", "w"))
    json.dump(rev_vocab, open(f"{vocab_dir}/rev_vocab.json", "w"))
