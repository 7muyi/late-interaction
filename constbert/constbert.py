import torch
import torch.nn as nn
from transformers import AutoModel

from utils import mv_score, maxsum


class ConstBERT(nn.Module):
    def __init__(self, pretrained_model: str, dim: int, vectors_per_passage: int, doc_maxlen: int) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)

        self.C = vectors_per_passage
        self.doc_maxlen = doc_maxlen

        self.W = nn.Parameter(torch.empty(self.doc_maxlen, self.C))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.W)

    def const_pooling(self, encode_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tok_repr = encode_outputs["tok_repr"]

        B, L, D = tok_repr.size()

        if L < self.doc_maxlen:
            tok_repr = nn.functional.pad(tok_repr, (0, 0, 0, self.doc_maxlen - L))

        pooled_repr = torch.einsum('bld,lc->bcd', tok_repr, self.W)

        pooled_repr = nn.functional.normalize(pooled_repr, p=2, dim=-1)
        pooled_mask = torch.ones(B, self.C, device=tok_repr.device)
        return {
            "tok_repr": pooled_repr,
            "tok_mask": pooled_mask
        }

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        tok_repr = self.proj(outputs)  # B, L, D
        tok_repr = torch.nn.functional.normalize(tok_repr, p=2, dim=-1)
        tok_repr = tok_repr * attention_mask.unsqueeze(-1)

        return {
            "tok_repr": tok_repr,
            "tok_mask": attention_mask
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encode_outputs = self.encode(input_ids, attention_mask)
        return self.const_pooling(encode_outputs)

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return maxsum(mv_score(qry_repr["tok_repr"], doc_repr["tok_repr"], pairwise))

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return ConstBERT.score(Q, D, False)