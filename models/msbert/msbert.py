import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import scorer  # register scorer functions
from common.registry import registry
from models.base_model import BaseModel, BaseEncoder


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


class PerSpanAttention(nn.Module):
    """SpanAttention with per-span query vectors (B, N, H) instead of a shared (B, H) query."""

    def __init__(self, hidden_size: int, n_heads: int, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_size, n_heads * dim)
        self.kv_proj = nn.Linear(hidden_size, 2 * n_heads * dim)
        self.out_proj = nn.Linear(n_heads * dim, hidden_size)

        for module in [self.q_proj, self.kv_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor, span_size: int) -> torch.Tensor:
        B, L, D = kv.shape
        assert L % span_size == 0, f"Length {L} must be divisible by span_size {span_size}"
        N = L // span_size

        q = self.q_proj(q).view(B, N, self.n_heads, self.dim).reshape(B * N, self.n_heads, 1, self.dim)

        kv = self.kv_proj(kv.reshape(B * N, span_size, D)).view(B * N, span_size, 2, self.n_heads, self.dim)
        k = kv[:, :, 0].transpose(1, 2).contiguous()
        v = kv[:, :, 1].transpose(1, 2).contiguous()

        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask.view(B * N, 1, 1, span_size).to(dtype=torch.bool),
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        return self.out_proj(attn_out.view(B, N, self.n_heads * self.dim))


@registry.register_model_name("msbert")
class MSBert(BaseModel, BaseEncoder):
    def __init__(
        self,
        pretrained_model: str,
        qry_span_size: int,
        doc_span_size: int,
        out_dim: int,
        attn_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size

        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.cls_proj = nn.Linear(self.llm.config.hidden_size, out_dim)
        self.span_attention = SpanAttention(
            self.llm.config.hidden_size,
            n_heads,
            attn_dim,
            dropout
        )
        self.span_proj = nn.Linear(self.llm.config.hidden_size, out_dim)

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
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

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

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor, torch.Tensor], D: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        return self.score(Q, D, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_span_size = config.get("qry_span_size", 4)
        doc_span_size = config.get("doc_span_size", 4)
        out_dim = config.get("out_dim", 128)
        attn_dim = config.get("attn_dim", 128)
        n_heads = config.get("n_heads", 8)
        dropout = config.get("dropout", 0.1)

        return cls(pretrained_model, qry_span_size, doc_span_size, out_dim, attn_dim, n_heads, dropout)


@registry.register_model_name("msbert_cls_prob")
class MSBert_ABL_CLS(BaseModel, BaseEncoder):
    def __init__(
        self,
        pretrained_model: str,
        qry_span_size: int,
        doc_span_size: int,
        out_dim: int,
        attn_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size

        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.span_attention = SpanAttention(
            self.llm.config.hidden_size,
            n_heads,
            attn_dim,
            dropout
        )
        self.span_proj = nn.Linear(self.llm.config.hidden_size, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.span_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_size: int | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        attn_out = self.span_attention(
            outputs[:, 0],  # [CLS] as query
            outputs[:, 2:-1],  # content tokens as kv
            attention_mask[:, 2:-1],
            span_size
        )  # B, N, H

        qd = outputs[:, 1].unsqueeze(1)  # B, 1, H
        combined = torch.cat([qd, attn_out], dim=1)  # B, N+1, H
        span_repr = F.normalize(self.span_proj(combined), p=2, dim=-1)

        content_mask = attention_mask[:, 2:-1].view(attention_mask.size(0), -1, span_size).any(-1)
        qd_mask = attention_mask[:, 1].unsqueeze(1).bool()
        span_mask = torch.cat([qd_mask, content_mask], dim=1)
        span_repr = span_repr * span_mask.unsqueeze(-1)

        return {"mv_repr": span_repr, "mv_mask": span_mask}

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.qry_span_size)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.doc_span_size)

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor, torch.Tensor], D: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        return self.score(Q, D, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_span_size = config.get("qry_span_size", 4)
        doc_span_size = config.get("doc_span_size", 4)
        out_dim = config.get("out_dim", 128)
        attn_dim = config.get("attn_dim", 128)
        n_heads = config.get("n_heads", 8)
        dropout = config.get("dropout", 0.1)

        return cls(pretrained_model, qry_span_size, doc_span_size, out_dim, attn_dim, n_heads, dropout)


@registry.register_model_name("msbert_qd_prob")
class MSBert_ABL_PROB(BaseModel, BaseEncoder):
    def __init__(
        self,
        pretrained_model: str,
        qry_span_size: int,
        doc_span_size: int,
        out_dim: int,
        attn_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size

        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.span_attention = SpanAttention(
            self.llm.config.hidden_size,
            n_heads,
            attn_dim,
            dropout
        )
        self.span_proj = nn.Linear(self.llm.config.hidden_size, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.span_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_size: int | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

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
            "mv_repr": span_repr,
            "mv_mask": span_mask,
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.qry_span_size)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.doc_span_size)

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor, torch.Tensor], D: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        return self.score(Q, D, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_span_size = config.get("qry_span_size", 4)
        doc_span_size = config.get("doc_span_size", 4)
        out_dim = config.get("out_dim", 128)
        attn_dim = config.get("attn_dim", 128)
        n_heads = config.get("n_heads", 8)
        dropout = config.get("dropout", 0.1)

        return cls(pretrained_model, qry_span_size, doc_span_size, out_dim, attn_dim, n_heads, dropout)



@registry.register_model_name("msbert_per_span_qd")
class MSBert_PerSpanQD(BaseModel, BaseEncoder):
    def __init__(
        self,
        pretrained_model: str,
        qry_span_size: int,
        doc_span_size: int,
        out_dim: int,
        attn_dim: int,
        n_heads: int,
        dropout: float,
        qry_n_qd_tokens: int = 8,
        doc_n_qd_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size
        self.qry_n_qd_tokens = qry_n_qd_tokens
        self.doc_n_qd_tokens = doc_n_qd_tokens

        self.llm = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.llm.config.hidden_size
        self.per_span_attention = PerSpanAttention(hidden_size, n_heads, attn_dim, dropout)
        self.span_proj = nn.Linear(hidden_size, out_dim)

        for module in [self.span_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_size: int,
        n_qd_tokens: int,
    ) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H
        B = outputs.shape[0]

        content = outputs[:, 1 + n_qd_tokens:-1]  # B, N*span_size, H
        cont_mask = attention_mask[:, 1 + n_qd_tokens:-1]
        N = content.shape[1] // span_size
        qd_query = outputs[:, 1:1 + N]  # B, N, H — each QD token queries its corresponding span

        attn_out = self.per_span_attention(qd_query, content, cont_mask, span_size)
        span_repr = F.normalize(self.span_proj(attn_out), p=2, dim=-1)
        span_mask = cont_mask.view(B, -1, span_size).any(-1)
        span_repr = span_repr * span_mask.unsqueeze(-1)

        return {"mv_repr": span_repr, "mv_mask": span_mask}

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.qry_span_size, self.qry_n_qd_tokens)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask, self.doc_span_size, self.doc_n_qd_tokens)

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
            attn_dim=config.get("attn_dim", 128),
            n_heads=config.get("n_heads", 8),
            dropout=config.get("dropout", 0.1),
            qry_n_qd_tokens=config.get("qry_n_qd_tokens", 8),
            doc_n_qd_tokens=config.get("doc_n_qd_tokens", 64),
        )
