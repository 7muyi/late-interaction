from dataclasses import dataclass

from base.config import BaseConfig


@dataclass
class Config(BaseConfig):
    # Model
    pretrained_model: str = "bert-base-uncased"
    n_heads: int = 8
    attn_dim: int = 128
    out_dim: int = 128
    qry_span_size: int = 4
    doc_span_size: int = 4
    dropout: float = 0.1

    # Tokenizer
    qry_maxlen: int = 32
    doc_maxlen: int = 256
    # qry_span_size: int = 4
    # doc_span_size: int = 4

    # Training
    dataset_path: str = ""
    checkpoint_path: str = ""

    amp: bool = True
    epoch: int = 5
    bsize: int = 128
    accumulation_steps: int = 2
    warmup: int = 1000
    lr_backbone: float = 3e-6
    lr_other: float = 1e-5
    lr_min_ratio: float = 0.1

    log_interval: int = 25

    temperature: float = 0.1