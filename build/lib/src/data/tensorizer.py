import torch


class Tensorizer:
    def __init__(self, vocab_size, context_length, output):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.output = output

    def tensorize(self, tokens):
        x = []
        y = []
        temp = torch.zeros(self.context_length, dtype=torch.int64)

        for i in range(len(tokens) - 1):
            x.append(temp)
            temp = temp[1:]
            temp = torch.cat((temp, torch.tensor([tokens[i]], dtype=torch.int64)))
            y.append(temp)

        x = torch.stack(x).to(torch.int64)
        y = torch.stack(y).to(torch.int64)
        return x, y
