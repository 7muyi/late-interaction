import torch

from common.registry import registry
from .base_tokenizer import BaseTokenizer
from .utils import insert_prefix_token_id


@registry.register_tokenizer_name("std_toker")
class StdTokenizer(BaseTokenizer):
    """
    Standard tokenizer with optional [Q]/[D] prefix.
    Format: [CLS] ([Q]/[D]) *tokens [SEP]
    """
    def _tokenize(self, texts: list[str], max_length: int, special_token: str | None) -> list[list[str]]:
        prefix = [self.cls_token, special_token] if special_token else [self.cls_token]
        return [
            prefix + self.tok.tokenize(text)[: max_length - (3 if self.use_prefix else 2)] + [self.sep_token]
            for text in texts
        ]

    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.qry_maxlen,
            self.Q_token if self.use_prefix else None
        )

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.doc_maxlen,
            self.D_token if self.use_prefix else None
        )

    def _tensorize(self, texts: list[str], max_length: int, special_token_id: int | None) -> tuple[torch.Tensor, ...]:
        if special_token_id is not None:
            obj = self.tok(texts, padding="longest", truncation="longest_first", return_tensors="pt", max_length=max_length - 1)

            ids = insert_prefix_token_id(obj["input_ids"], special_token_id)
            masks = insert_prefix_token_id(obj["attention_mask"], 1)

            return ids, masks

        obj = self.tok(texts, padding="longest", truncation="longest_first", return_tensors="pt", max_length=max_length)
        return obj["input_ids"], obj["attention_mask"]

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.qry_maxlen,
            self.Q_token_id if self.use_prefix else None
        )

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.doc_maxlen,
            self.D_token_id if self.use_prefix else None
        )

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 256)
        use_prefix = config.get("use_prefix", True)

        return cls(pretrained_model, qry_maxlen, doc_maxlen, use_prefix)
