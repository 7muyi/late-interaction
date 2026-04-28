import torch
import torch.nn as nn
from transformers import AutoModel

import scorer  # register scorer functions
from common.registry import registry
from models.base_model import BaseModel, BaseEncoder


@registry.register_model_name("tokenpruning")
class TokenPruning(BaseModel, BaseEncoder):
    def __init__(self, pretrained_model: str, dim: int, ns_dim: int, norm_threshold: float) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim, bias=False)
        self.projector = nn.Linear(self.llm.config.hidden_size, ns_dim, bias=False)
        self.norm_threshold = norm_threshold

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.orthogonal_(self.projector.weight, gain=1e-6)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        base_repr = self.proj(outputs)
        proj_repr = self.projector(outputs)

        combined_norm = torch.sqrt(base_repr.norm(dim=-1, keepdim=True)**2 + proj_repr.norm(dim=-1, keepdim=True)**2)
        tok_repr = base_repr / combined_norm.clamp(min=1e-9)
        tok_repr = tok_repr * attention_mask.unsqueeze(-1)

        return {
            "mv_repr": tok_repr,
            "mv_mask": attention_mask
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        res = self.encode(input_ids, attention_mask)
        tok_repr = res["mv_repr"]

        norm_mask = tok_repr.norm(dim=-1) >= self.norm_threshold
        attention_mask = torch.logical_and(norm_mask, attention_mask)
        tok_repr = tok_repr * attention_mask.unsqueeze(-1)

        return {
            "mv_repr": tok_repr,
            "mv_mask": attention_mask
        }

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode(*Q)
        D = self.encode(*D)

        return self.score(Q, D, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        ns_dim = config.get("ns_dim", 32)
        norm_threshold = config.get("norm_threshold", 0.5)

        return cls(pretrained_model, dim, ns_dim, norm_threshold)