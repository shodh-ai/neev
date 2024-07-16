import torch
import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, targets):
        return super().forward(logits, targets)


class ChunkedCrossEntropyLoss(nn.Module):
    def __init__(self, chunk_size=128, ignore_index=-100):
        super().__init__()
        self.subsidary_loss = nn.CrossEntropyLoss()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if isinstance(logits, list):
            # don't want to chunk cross entropy
            if self.chunk_size == 0:
                logits = torch.cat(logits, dim=0)  # Concatenate along the batch dimension
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
                return torch.nn.functional.cross_entropy(
                    logits, targets, ignore_index=self.ignore_index
                )
            
            # chunk cross entropy
            logit_chunks = [
                logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
            ]
            target_chunks = [
                target_chunk.reshape(-1)
                for target_chunk in targets.split(1, dim=1)
            ]
            loss_chunks = [
                torch.nn.functional.cross_entropy(
                    logit_chunk,
                    target_chunk,
                    ignore_index=self.ignore_index,
                    reduction="none",
                )
                for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
            ]
            non_masked_elems = (targets != self.ignore_index).sum()
            return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(
                torch.ones_like(non_masked_elems)
            )

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        if self.chunk_size == 0:
            return torch.nn.functional.cross_entropy(
                logits, targets, ignore_index=self.ignore_index
            )
        # lm_head wasn't chunked, chunk cross entropy
        logit_chunks = logits.split(self.chunk_size)
        target_chunks = targets.split(self.chunk_size)
        loss_chunks = [
            torch.nn.functional.cross_entropy(
                logit_chunk,
                target_chunk,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != self.ignore_index).sum()
        # [non_masked_elems div note]:
        #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
        #   results in a python int which is then passed back to torch division. By using the
        #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
        return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(
            torch.ones_like(non_masked_elems)
        )
