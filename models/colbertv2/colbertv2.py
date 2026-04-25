import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import scorer  # register scorer functions
from common.registry import registry
from models.base_model import BaseModel, BaseEncoder


@registry.register_model_name("colbertv2")
class ColBERTv2(BaseModel, BaseEncoder):
    def __init__(self, pretrained_model: str, dim: int, sim_log_temp: float = 1.0, topk: int = 32) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)
        self.sim_log_temp = nn.Parameter(torch.tensor(math.log(sim_log_temp)))
        self.topk = topk

        self._init_weights()

    @property
    def sim_temperature(self) -> torch.Tensor:
        return self.sim_log_temp.exp().clamp(min=0.01, max=1)

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

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("soft_maxsim_sum")(qry_repr, doc_repr, pairwise=pairwise, topk=self.topk, temperature=self.sim_temperature)

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        return self.score(Q, D, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        sim_log_temp = config.get("sim_log_temp", 1.0)
        topk = config.get("topk", 32)

        return cls(pretrained_model, dim, sim_log_temp, topk)