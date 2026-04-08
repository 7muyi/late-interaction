import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from utils import mv_score, maxsum, sv_score
from .config import Config


class SpanAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_size, n_heads * dim)
        self.kv_proj = nn.Linear(hidden_size, 2 * n_heads * dim)
        self.out_proj = nn.Linear(n_heads * dim, hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.q_proj, self.kv_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor, span_size: int) -> torch.Tensor:
        B, L, D = kv.shape
        assert L % span_size == 0, f"Length {L} must be divisible by span_size {span_size}"
        N = L // span_size

        q_shared = self.q_proj(q).view(B, 1, self.n_heads, self.dim)
        q_shared = q_shared.expand(B, N, self.n_heads, self.dim).reshape(B * N, self.n_heads, 1, self.dim)

        kv = self.kv_proj(kv.reshape(B * N, span_size, D)).view(B * N, span_size, 2, self.n_heads, self.dim)

        k = kv[:, :, 0].transpose(1, 2).contiguous()
        v = kv[:, :, 1].transpose(1, 2).contiguous()

        final_mask = mask.view(B * N, 1, 1, span_size).to(dtype=torch.bool)

        attn_out = F.scaled_dot_product_attention(
            query=q_shared,
            key=k,
            value=v,
            attn_mask=final_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        attn_out = attn_out.view(B, N, self.n_heads * self.dim)

        return self.out_proj(attn_out)


class MSBert(BaseModel):
    def __init__(
        self,
        config: Config
    ) -> None:
        super().__init__()

        self.qry_span_size = config.qry_span_size
        self.doc_span_size = config.doc_span_size

        self.llm = AutoModel.from_pretrained(config.pretrained_model)
        self.cls_proj = nn.Linear(self.llm.config.hidden_size, config.out_dim)
        self.span_attention = SpanAttention(
            self.llm.config.hidden_size,
            config.n_heads,
            config.attn_dim,
            config.dropout
        )
        self.span_proj = nn.Linear(self.llm.config.hidden_size, config.out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.cls_proj, self.span_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_size: int | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask)[0]  # B, L, H

        cls_repr = self.cls_proj(outputs[:, 0])  # B, D
        cls_repr = F.normalize(cls_repr, p=2, dim=-1)

        span_repr = self.span_attention(
            outputs[:, 1],
            outputs[:, 2:-1],
            attention_mask[:, 2:-1],
            span_size
        )
        span_repr = self.span_proj(span_repr)  # B, N, H
        span_repr = F.normalize(span_repr, p=2, dim=-1)
        span_mask = attention_mask[:, 2:-1].view(attention_mask.size(0), -1, span_size).any(-1)
        span_repr = span_repr * span_mask.unsqueeze(-1)

        return {
            "cls_repr": cls_repr,
            "mv_repr": span_repr,
            "mv_mask": span_mask,
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.qry_span_size)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.doc_span_size)

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        cls_scores = sv_score(qry_repr["cls_repr"], doc_repr["cls_repr"], pairwise)
        span_scores = maxsum(mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise))

        if "mv_mask" in qry_repr:
            span_scores = span_scores / qry_repr["mv_mask"].sum(-1, keepdim=True)

        return cls_scores + span_scores

    def forward(self, Q: tuple[torch.Tensor, torch.Tensor], D: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        return MSBert.score(Q, D, False)