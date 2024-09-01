import math
import torch
import torch.nn as nn


class Learned(nn.Module):
    def __init__(self, contextLength, embeddingDim, dtype):
        super(Learned, self).__init__()

        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dtype = dtype

        self.pos_embedding_layer = nn.Embedding(
            self.contextLength, self.embeddingDim, dtype=self.dtype
        )

    def forward(self, x):
        pos_indices = torch.arange(self.contextLength, device=x.device)
        pos_embedding = self.pos_embedding_layer(pos_indices).to(self.dtype)
        x = x + pos_embedding
        return x


class Cosine(nn.Module):
    def __init__(self, contextLength, embeddingDim, dtype):
        super(Cosine, self).__init__()

        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dtype = dtype

        self.posVector = torch.zeros(
            self.contextLength, self.embeddingDim, dtype=self.dtype
        )
        for pos in range(self.contextLength):
            for i in range(0, self.embeddingDim, 2):
                self.posVector[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i) / self.embeddingDim))
                )
                self.posVector[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * i) / self.embeddingDim))
                )
        self.posVector = self.posVector.to(self.dtype)

    def forward(self, x):
        x = x + self.posVector.to(x.device)
        return x


class RoPE(nn.Module):
    def __init__(
        self, context_length, embedding_dim, dtype, base=10000, save_inv_freqs=False
    ):
        super(RoPE, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, embedding_dim, 2) / embedding_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.dim = embedding_dim
        self.context_length = context_length
        self.dtype = dtype
        self.base = base

        self.context_length_cached = None
        self.cos_cached = None
        self.sin_cached = None

        cos_cached, sin_cached, inv_freq = self._prepare_cache()

        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def _prepare_cache(self):
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        t = torch.arange(self.context_length).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_cached = emb.cos()[:, None, None, :]
        sin_cached = emb.sin()[:, None, None, :]

        return (
            cos_cached.to(self.dtype),
            sin_cached.to(self.dtype),
            inv_freq.to(self.dtype),
        )

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        assert seq_len <= self.context_length

        if seq_len != self.context_length:
            return (
                self.cos_cached[:seq_len, ...].to(x.device),
                self.sin_cached[:seq_len, ...].to(x.device),
            )
        else:
            return self.cos_cached.to(x.device), self.sin_cached.to(x.device)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=x1.ndim - 1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, offset: int = 0):
        cos, sin = (
            cos[offset : q.shape[0] + offset, ...],
            sin[offset : q.shape[0] + offset, ...],
        )

        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (
            self.rotate_half(k) * sin
        )

    def apply_rotary_pos_emb_torch(self, q, k, cos, sin, offset: int = 0):
        cos, sin = (
            cos[offset : q.shape[0] + offset, ...],
            sin[offset : q.shape[0] + offset, ...],
        )
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (
            self.rotate_half(k) * sin
        )
