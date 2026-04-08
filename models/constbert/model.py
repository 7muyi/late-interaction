import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from utils import mv_score, maxsum
from .config import Config


class ConstBERT(BaseModel):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(config.pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, config.dim)

        self.C = config.vectors_per_passage
        self.doc_maxlen = config.doc_maxlen

        self.W = nn.Parameter(torch.empty(self.doc_maxlen, self.C))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.W)

    def const_pooling(self, encode_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tok_repr = encode_outputs["mv_repr"]
        tok_mask = encode_outputs["mv_mask"]

        B, L, D = tok_repr.size()

        if L < self.doc_maxlen:
            tok_repr = tok_repr * tok_mask.unsqueeze(-1)
            tok_repr = F.pad(tok_repr, (0, 0, 0, self.doc_maxlen - L))

        pooled_repr = torch.einsum('bld,lc->bcd', tok_repr, self.W)

        pooled_repr = F.normalize(pooled_repr, p=2, dim=-1)
        pooled_mask = torch.ones(B, self.C, device=tok_repr.device)
        return {
            "mv_repr": pooled_repr,
            "mv_mask": pooled_mask
        }

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        tok_repr = self.proj(outputs)  # B, L, D
        tok_repr = F.normalize(tok_repr, p=2, dim=-1)
        tok_repr = tok_repr * attention_mask.unsqueeze(-1)

        return {
            "mv_repr": tok_repr,
            "mv_mask": attention_mask
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encode_outputs = self.encode(input_ids, attention_mask)
        return self.const_pooling(encode_outputs)

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return maxsum(mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise))

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return ConstBERT.score(Q, D, False)