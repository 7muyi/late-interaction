import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import scorer  # register scorer functions
from common.registry import registry
from models.base_model import BaseModel, BaseEncoder


@registry.register_model_name("constbert")
class ConstBERT(BaseModel, BaseEncoder):
    def __init__(self, pretrained_model: str, dim: int, doc_maxlen: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.doc_maxlen = doc_maxlen
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)
        self.doc_project = nn.Linear(doc_maxlen, dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

        self._init_weights()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=0.1)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.doc_project.weight)
        nn.init.zeros_(self.doc_project.bias)

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        Q = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.proj(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)
        Q = Q * attention_mask.unsqueeze(-1)

        return {
            "mv_repr": Q,
            "mv_mask": attention_mask
        }

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        D = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        D = D.permute(0, 2, 1)  # B, H, L
        D = self.doc_project(D)  # B, H, C
        D = D.permute(0, 2, 1)  # B, C, H
        D = self.proj(D)
        D = F.normalize(D, p=2, dim=2)

        mask = torch.ones(D.shape[0], D.shape[1], device=D.device, dtype=attention_mask.dtype)

        return {
            "mv_repr": D,
            "mv_mask": mask
        }

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return self.score(Q, D, False) / self.temperature

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        doc_maxlen = config.get("doc_maxlen", 32)
        temperature = config.get("temperature", 1.0)

        return cls(pretrained_model, dim, doc_maxlen, temperature)