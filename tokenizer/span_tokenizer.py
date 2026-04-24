import torch

import numpy as np

from common.registry import registry
from .base_tokenizer import BaseTokenizer


@registry.register_tokenizer_name("span_toker")
class SpanTokenizer(BaseTokenizer):
    """
    Span-aligned tokenizer with optional [Q]/[D] prefix.
    Pads token count (excluding specials) to a multiple of span_size.
    Format: [CLS] ([Q]/[D]) tokens [PAD...] [SEP]
    """

    def __init__(self, pretrained_model: str, qry_maxlen: int,
                 doc_maxlen: int, qry_span_size: int, doc_span_size: int,
                 use_prefix: bool = True) -> None:
        super().__init__(pretrained_model, qry_maxlen, doc_maxlen, use_prefix)

        self.qry_span_size = qry_span_size
        self.doc_span_size = doc_span_size

        self.n_special = 3 if self.use_prefix else 2
        self.qry_maxlen = (qry_maxlen - self.n_special) // self.qry_span_size * self.qry_span_size + self.n_special
        self.doc_maxlen = (doc_maxlen - self.n_special) // self.doc_span_size * self.doc_span_size + self.n_special

    def _tokenize(
        self,
        texts: list[str],
        span_size: int,
        max_length: int,
        special_token: str | None
    ) -> list[list[str]]:
        prefix = [self.cls_token, special_token] if special_token else [self.cls_token]
        results = []
        for text in texts:
            tokens = self.tok.tokenize(text)[: max_length - self.n_special]
            pad_count = (span_size - (len(tokens) % span_size)) % span_size
            results.append(prefix + tokens + [self.pad_token] * pad_count + [self.sep_token])
        return results

    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.qry_span_size,
            self.qry_maxlen,
            self.Q_token if self.use_prefix else None
        )

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.doc_span_size,
            self.doc_maxlen,
            self.D_token if self.use_prefix else None
        )

    def _tensorize(
        self,
        texts: list[str],
        span_size: int,
        max_length: int,
        special_token_id: int | None
    ) -> tuple[torch.Tensor, ...]:
        all_ids = [
            self.tok.encode(text, add_special_tokens=False, max_length=max_length - self.n_special, truncation=True)
            for text in texts
        ]
        maxlen = max(len(ids) for ids in all_ids)
        maxlen = (maxlen + span_size - 1) // span_size * span_size + self.n_special

        ids = np.full((len(all_ids), maxlen), self.pad_token_id, dtype=np.int64)
        masks = np.zeros_like(ids, dtype=np.int64)

        ids[:, 0] = self.cls_token_id
        if special_token_id is not None:
            ids[:, 1] = special_token_id
            content_offset = 2
        else:
            content_offset = 1

        for i, tok_ids in enumerate(all_ids):
            seq_len = min(len(tok_ids), maxlen - self.n_special)
            ids[i, content_offset: content_offset + seq_len] = tok_ids
            ids[i, content_offset + seq_len] = self.sep_token_id
            masks[i, :content_offset + seq_len + 1] = 1

        return torch.from_numpy(ids), torch.from_numpy(masks)

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.qry_span_size,
            self.qry_maxlen,
            self.Q_token_id if self.use_prefix else None
        )

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.doc_span_size,
            self.doc_maxlen,
            self.D_token_id if self.use_prefix else None
        )

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 256)
        qry_span_size = config.get("qry_span_size", 4)
        doc_span_size = config.get("doc_span_size", 4)
        use_prefix = config.get("use_prefix", True)

        return cls(pretrained_model, qry_maxlen, doc_maxlen, qry_span_size, doc_span_size, use_prefix)
