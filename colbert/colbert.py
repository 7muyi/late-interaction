import torch
import torch.nn as nn
from transformers import AutoModel

from utils import mv_score, maxsum


class ColBERT(nn.Module):
    def __init__(self, pretrained_model: str, dim: int) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        tok_repr = self.proj(outputs)  # B, L, D
        tok_repr = torch.nn.functional.normalize(tok_repr, p=2, dim=-1)
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
        return maxsum(mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise))

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return ColBERT.score(Q, D, False)