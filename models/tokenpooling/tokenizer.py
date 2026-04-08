import torch

from base.tokenizer import BaseTokenizer
from utils import insert_prefix_token_id


class Tokenizer(BaseTokenizer):
    def _tokenize(
        self,
        texts: list[str],
        max_length: int,
        special_token: str
    ) -> list[list[str]]:
        """Tokenizes texts into lists of tokens with special markers: [CLS], [Q]/[D], *tokenized_text*, [SEP]"""
        return [
            [self.cls_token, special_token] + self.tok.tokenize(text)[: max_length - 3] + [self.sep_token]
            for text in texts
        ]

    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.qry_maxlen,
            self.Q_token
        )

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        return self._tokenize(
            texts,
            self.doc_maxlen,
            self.D_token
        )

    def _tensorize(
        self,
        texts: list[str],
        max_length: int,
        special_token_id: int
    ) -> tuple[torch.Tensor, ...]:
        """Converts texts into padded tensors of token ids and attention masks."""
        obj = self.tok(texts, padding="longest", truncation="longest_first", return_tensors="pt", max_length=max_length - 1)

        ids = insert_prefix_token_id(obj["input_ids"], special_token_id)
        masks = insert_prefix_token_id(obj["attention_mask"], 1)

        return ids, masks

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.qry_maxlen,
            self.Q_token_id
        )

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        return self._tensorize(
            texts,
            self.doc_maxlen,
            self.D_token_id
        )
