import torch

from common.registry import registry
from .base_tokenizer import BaseTokenizer
from .utils import insert_prefix_token_id


@registry.register_tokenizer_name("const_toker")
class StdTokenizer(BaseTokenizer):
    """
    ConstBERT's tokenizer with optional [Q]/[D] prefix.
    Format: [CLS] ([Q]/[D]) *tokens [PAD] ... [SEP] (Fixed length)
    """
    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        prefix = [self.cls_token, self.Q_token] if self.use_prefix else [self.cls_token]
        return [
            prefix + self.tok.tokenize(text)[: self.qry_maxlen - (3 if self.use_prefix else 2)] + [self.sep_token]
            for text in texts
        ]

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        prefix = [self.cls_token, self.D_token] if self.use_prefix else [self.cls_token]
        results = []
        for text in texts:
            toks = self.tok.tokenize(text)[: self.doc_maxlen - (3 if self.use_prefix else 2)]
            results.append(
                prefix + toks + (self.doc_maxlen - len(toks) - len(prefix)) * [self.pad_token] + [self.sep_token]
            )
        return results

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        if self.use_prefix:
            obj = self.tok(texts, padding="longest", truncation="longest_first", return_tensors="pt", max_length=self.qry_maxlen - 1)

            ids = insert_prefix_token_id(obj["input_ids"], self.Q_token_id)
            masks = insert_prefix_token_id(obj["attention_mask"], 1)

            return ids, masks

        obj = self.tok(texts, padding="longest", truncation="longest_first", return_tensors="pt", max_length=self.qry_maxlen)
        return obj["input_ids"], obj["attention_mask"]

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        if self.use_prefix:
            obj = self.tok(texts, padding="max_length", truncation="longest_first", return_tensors="pt", max_length=self.doc_maxlen - 1)

            ids = insert_prefix_token_id(obj["input_ids"], self.D_token_id)
            masks = insert_prefix_token_id(obj["attention_mask"], 1)

            return ids, masks

        obj = self.tok(texts, padding="max_length", truncation="longest_first", return_tensors="pt", max_length=self.doc_maxlen)
        return obj["input_ids"], obj["attention_mask"]

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 256)
        use_prefix = config.get("use_prefix", True)

        return cls(pretrained_model, qry_maxlen, doc_maxlen, use_prefix)
