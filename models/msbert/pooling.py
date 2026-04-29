import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from common.registry import registry
from models.base_model import BaseModel, BaseEncoder


class MeanSpanPooling(nn.Module):
    # nn.AvgPool1d divides by kernel_size, not valid token count — use masked mean
    def forward(self, x: torch.Tensor, mask: torch.Tensor, span_size: int) -> torch.Tensor:
        B, L, H = x.shape
        N = L // span_size
        x = x.view(B, N, span_size, H)
        m = mask.view(B, N, span_size, 1).float()
        return (x * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)


class MaxSpanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor, span_size: int) -> torch.Tensor:
        B, L, H = x.shape
        N = L // span_size
        x = x.view(B, N, span_size, H)
        m = mask.view(B, N, span_size, 1).bool()
        out = x.masked_fill(~m, float("-inf")).max(dim=2).values  # B, N, H
        # All-padding spans produce -inf; zero them to avoid NaN in downstream linear layers
        return out.masked_fill(~m.any(dim=2), 0.0)


class LinearSpanPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, span_size: int) -> torch.Tensor:
        B, L, H = x.shape
        N = L // span_size
        x = x.view(B, N, span_size, H)
        m = mask.view(B, N, span_size).bool()
        scores = self.score(x).squeeze(-1).masked_fill(~m, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * x).sum(dim=2)


class _MSBertPoolBase(BaseModel, BaseEncoder):
    def __init__(
        self,
        pretrained_model: str,
        qry_span_size: int,
        doc_span_size: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size

        self.llm = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.llm.config.hidden_size

        self.span_pooling = self._build_pooling(hidden_size)
        self.span_proj = nn.Linear(hidden_size, out_dim)
        nn.init.xavier_uniform_(self.span_proj.weight)
        nn.init.zeros_(self.span_proj.bias)

    def _build_pooling(self, hidden_size: int) -> nn.Module:
        raise NotImplementedError

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_size: int,
    ) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        tokens = outputs[:, 2:-1]
        token_mask = attention_mask[:, 2:-1]

        pooled = self.span_pooling(tokens, token_mask, span_size)  # B, N, H

        span_repr = self.span_proj(pooled)
        span_repr = F.normalize(span_repr, p=2, dim=-1)

        span_mask = token_mask.view(token_mask.size(0), -1, span_size).any(-1)  # B, N
        span_repr = span_repr * span_mask.unsqueeze(-1)

        return {"mv_repr": span_repr, "mv_mask": span_mask}

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.qry_span_size)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.doc_span_size)

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor, torch.Tensor], D: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.score(self.encode_qry(*Q), self.encode_doc(*D), False)

    @classmethod
    def from_config(cls, config):
        return cls(
            pretrained_model=config.get("pretrained_model", "bert-base-uncased"),
            qry_span_size=config.get("qry_span_size", 4),
            doc_span_size=config.get("doc_span_size", 4),
            out_dim=config.get("out_dim", 128),
        )


@registry.register_model_name("msbert_mean_pool")
class MSBert_MeanPool(_MSBertPoolBase):
    def _build_pooling(self, hidden_size: int) -> nn.Module:
        return MeanSpanPooling()


@registry.register_model_name("msbert_max_pool")
class MSBert_MaxPool(_MSBertPoolBase):
    def _build_pooling(self, hidden_size: int) -> nn.Module:
        return MaxSpanPooling()


@registry.register_model_name("msbert_linear_pool")
class MSBert_LinearPool(_MSBertPoolBase):
    def _build_pooling(self, hidden_size: int) -> nn.Module:
        return LinearSpanPooling(hidden_size)
