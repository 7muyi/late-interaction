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
        self.doc_project = nn.Linear(config.doc_maxlen, config.dim)

        self._init_weights()

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

        mask = torch.ones(D.shape[0], D.shape[1], device=input_ids.device, dtype=attention_mask.dtype)

        return {
            "mv_repr": D,
            "mv_mask": mask
        }

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return maxsum(mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise))

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return ConstBERT.score(Q, D, False)