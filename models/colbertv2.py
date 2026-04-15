import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from common.registry import registry
from .base_model import BaseModel, BaseEncoder
from .utils import mv_score


@registry.register_model_name("colbertv2")
class ColBERTv2(BaseModel, BaseEncoder):
    def __init__(self, pretrained_model: str, dim: int, temperature: float = 1.0, topk: int = 32) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.topk = topk

        self._init_weights()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=0.5)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

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
        return self.encode(input_ids, attention_mask)

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        P = mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise)
        scores = P.max(dim=-1).values.sum(-1)

        if "mv_mask" in qry_repr:
            scores = scores / qry_repr["mv_mask"].sum(-1, keepdim=True)

        return scores

    @staticmethod
    def soft_maxsim(qry_repr: dict, doc_repr: dict, temperature: torch.Tensor, topk: int, pairwise: bool = False) -> torch.Tensor:
        # P: (..., n, m) similarity matrix between query and doc tokens
        P = mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise)
        # keep only top-k elements per row
        k = min(topk, P.size(-1))
        P = P.topk(k, dim=-1).values  # (..., n, k)
        # soft-max approximation of MaxSim: tau * log(sum_j exp(q_i d_j^T / tau))
        scores = temperature * torch.logsumexp(P / temperature, dim=-1)  # (..., n)
        scores = scores.sum(-1)  # (...)

        if "mv_mask" in qry_repr:
            scores = scores / qry_repr["mv_mask"].sum(-1, keepdim=True)

        return scores

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        # soft-maxsim for training
        # optional: maxsim for inference
        return ColBERTv2.soft_maxsim(Q, D, self.temperature, self.topk, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        temperature = config.get("temperature", 1.0)
        topk = config.get("topk", 32)

        return cls(pretrained_model, dim, temperature, topk)