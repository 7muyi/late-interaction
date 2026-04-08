import numpy as np
import torch

from base.tokenizer import BaseTokenizer
from .config import Config


class Tokenizer(BaseTokenizer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # pre-calculate the maximum number of tokens.
        # 3 tokens are reserved for special tokens: [CLS], [Q]/[D], and [SEP].
        # ensure that the number of valid tokens is a multiple of span_size.
        self.qry_span_size = config.qry_span_size
        self.doc_span_size = config.doc_span_size
        self.qry_maxlen = (config.qry_maxlen - 3) // self.qry_span_size * self.qry_span_size + 3
        self.doc_maxlen = (config.doc_maxlen - 3) // self.doc_span_size * self.doc_span_size + 3

    def _tokenize(
        self,
        texts: list[str],
        span_size: int,
        max_length: int,
        special_token: int
    ) -> list[list[str]]:
        """Tokenizes a text into a list of tokens with special markers: [CLS], [Q]/[D], *tokenized_text*, [SEP]"""
        results = []
        for text in texts:
            tokens = self.tok.tokenize(text)[: max_length - 3]
            results.append(
                [self.cls_token, special_token] +
                tokens +
                [self.pad_token] * ((span_size - (len(tokens) % span_size)) % span_size) +
                [self.sep_token]
            )
        return results

    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.qry_span_size,
            self.qry_maxlen,
            self.Q_token
        )

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.doc_span_size,
            self.doc_maxlen,
            self.D_token
        )

    def _tensorize(
        self,
        texts: list[str],
        span_size: int,
        max_length: int,
        special_token_id: int
    ):
        """
        Converts text(s) into padded tensors of token ids and attention masks.
        Ensures the sequence length (excluding special tokens) is a multiple of span_size.
        """
        all_ids = [
            self.tok.encode(
                text,
                add_special_tokens=False,
                max_length=max_length - 3,
                truncation=True
            )
            for text in texts
        ]
        maxlen = max(len(ids) for ids in all_ids)
        # ensure the number of valid tokens is a multiple of span_size while preserving as much information as possible.
        maxlen = (maxlen + span_size - 1) // span_size * span_size + 3  # +3 for [CLS], [Q]/[D], [SEP]

        # initialize
        ids = np.full((len(all_ids), maxlen), self.pad_token_id, dtype=np.int64)
        masks = np.zeros_like(ids, dtype=np.int64)

        # set special tokens
        ids[:, 0] = self.cls_token_id
        ids[:, 1] = special_token_id

        # populate data
        for i, tok_ids in enumerate(all_ids):
            seq_len = min(len(tok_ids), maxlen - 3)
            ids[i, 2: 2 + seq_len] = tok_ids
            ids[i, 2 + seq_len] = self.sep_token_id  # add [SEP] token
            masks[i, :seq_len + 3] = 1

        # convert to Tensor at Once
        ids = torch.from_numpy(ids)
        masks = torch.from_numpy(masks)

        return ids, masks

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.qry_span_size,
            self.qry_maxlen,
            self.Q_token_id
        )

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.doc_span_size,
            self.doc_maxlen,
            self.D_token_id
        )