import torch
from transformers import AutoTokenizer


class BaseTokenizer:
    def __init__(self, config) -> None:
        self.tok = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.qry_maxlen = config.qry_maxlen
        self.doc_maxlen = config.doc_maxlen

        # special tokens: [Q], [D], [SEP], [CLS], [PAD]
        self.Q_token = "[Q]"
        self.D_token = "[D]"
        self.sep_token = self.tok.sep_token
        self.cls_token = self.tok.cls_token
        self.pad_token = self.tok.pad_token

        # the id corresponding to the special token
        self.Q_token_id = self.tok.convert_tokens_to_ids("[unused0]")
        self.D_token_id = self.tok.convert_tokens_to_ids("[unused1]")
        self.sep_token_id = self.tok.sep_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.pad_token_id = self.tok.pad_token_id

    def tokenize_qry(self, texts: list[str]) -> list[list[str]]:
        raise NotImplementedError()

    def tokenize_doc(self, texts: list[str]) -> list[list[str]]:
        raise NotImplementedError()

    def tensorize_qry(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError()

    def tensorize_doc(self, texts: list[str]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError()
