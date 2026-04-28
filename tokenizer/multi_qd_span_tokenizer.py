import numpy as np
import torch

from common.registry import registry
from .base_tokenizer import BaseTokenizer


@registry.register_tokenizer_name("multi_qd_span_toker")
class MultiQDSpanTokenizer(BaseTokenizer):
    """
    Span-aligned tokenizer that inserts n_qd_tokens copies of [Q]/[D] after [CLS].
    The i-th [Q]/[D] is used as the query for the i-th span in PerSpanAttention.
    Format: [CLS] [QD]*n_qd_tokens tokens [PAD...] [SEP]
    Query and document may use different numbers of QD tokens.
    """

    def __init__(
        self,
        pretrained_model: str,
        qry_maxlen: int,
        doc_maxlen: int,
        qry_span_size: int,
        doc_span_size: int,
        qry_n_qd_tokens: int = 8,
        doc_n_qd_tokens: int = 64,
    ) -> None:
        super().__init__(pretrained_model, qry_maxlen, doc_maxlen, use_prefix=True)

        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size
        self.qry_n_qd_tokens = qry_n_qd_tokens
        self.doc_n_qd_tokens = doc_n_qd_tokens

        qry_n_special = 2 + qry_n_qd_tokens  # CLS + qry_n_qd_tokens × [Q] + SEP
        doc_n_special = 2 + doc_n_qd_tokens
        self.qry_maxlen = (qry_maxlen - qry_n_special) // qry_span_size * qry_span_size + qry_n_special
        self.doc_maxlen = (doc_maxlen - doc_n_special) // doc_span_size * doc_span_size + doc_n_special

    def _tokenize(
        self,
        texts: list[str],
        span_size: int,
        max_length: int,
        special_token: str,
        n_qd_tokens: int,
    ) -> list[list[str]]:
        n_special = 2 + n_qd_tokens
        prefix = [self.cls_token] + [special_token] * n_qd_tokens
        results = []
        for text in texts:
            tokens = self.tok.tokenize(text)[: max_length - n_special]
            pad_count = (span_size - len(tokens) % span_size) % span_size
            results.append(prefix + tokens + [self.pad_token] * pad_count + [self.sep_token])
        return results

    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(texts, self.qry_span_size, self.qry_maxlen, self.Q_token, self.qry_n_qd_tokens)

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(texts, self.doc_span_size, self.doc_maxlen, self.D_token, self.doc_n_qd_tokens)

    def _tensorize(
        self,
        texts: list[str],
        span_size: int,
        max_length: int,
        special_token_id: int,
        n_qd_tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        n_special = 2 + n_qd_tokens
        content_offset = 1 + n_qd_tokens
        all_ids = [
            self.tok.encode(text, add_special_tokens=False, max_length=max_length - n_special, truncation=True)
            for text in texts
        ]
        maxlen = max(len(ids) for ids in all_ids)
        maxlen = (maxlen + span_size - 1) // span_size * span_size + n_special

        ids = np.full((len(all_ids), maxlen), self.pad_token_id, dtype=np.int64)
        masks = np.zeros_like(ids, dtype=np.int64)

        ids[:, 0] = self.cls_token_id
        ids[:, 1 : 1 + n_qd_tokens] = special_token_id

        for i, tok_ids in enumerate(all_ids):
            seq_len = min(len(tok_ids), maxlen - n_special)
            ids[i, content_offset : content_offset + seq_len] = tok_ids[:seq_len]
            ids[i, content_offset + seq_len] = self.sep_token_id
            masks[i, : content_offset + seq_len + 1] = 1

        return torch.from_numpy(ids), torch.from_numpy(masks)

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(texts, self.qry_span_size, self.qry_maxlen, self.Q_token_id, self.qry_n_qd_tokens)

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(texts, self.doc_span_size, self.doc_maxlen, self.D_token_id, self.doc_n_qd_tokens)

    @classmethod
    def from_config(cls, config):
        return cls(
            pretrained_model=config.get("pretrained_model", "bert-base-uncased"),
            qry_maxlen=config.get("qry_maxlen", 32),
            doc_maxlen=config.get("doc_maxlen", 256),
            qry_span_size=config.get("qry_span_size", 4),
            doc_span_size=config.get("doc_span_size", 4),
            qry_n_qd_tokens=config.get("qry_n_qd_tokens", 8),
            doc_n_qd_tokens=config.get("doc_n_qd_tokens", 64),
        )
