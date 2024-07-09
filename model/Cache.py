import torch
from typing import Tuple
from torch import nn, Tensor

class KVCache(nn.Module):
    def __init__(
        self,
        maxBatchSize: int,
        contextLength: int,
        numHeads: int,
        headDim: int,
        dtype: str,
        device: torch.device
    ) -> None:
        super().__init__()
        
        dtype_mapping = {
            "64-true": torch.float64,
            "32-true": torch.float32,
            "16-true": torch.float16,
            "bf16": torch.bfloat16,
            64: torch.float64,
            32: torch.float32,
            16: torch.float16,
        }
        
        if dtype not in dtype_mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        torch_dtype = dtype_mapping[dtype]

        cache_shape = (maxBatchSize, numHeads, contextLength, headDim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=torch_dtype, device=device), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=torch_dtype, device=device), persistent=False
        )
        self.maxBatchSize = maxBatchSize

    def get_cache(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        return self.k_cache[layer_id], self.v_cache[layer_id]

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor) -> Tuple[Tensor, Tensor]:
        assert input_pos.shape[0] == k_val.shape[2]

        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val

        return self.k_cache, self.v_cache
