from dataclasses import dataclass


@dataclass
class Config:
    # ColBERT parameters
    pretrained_model: str = "bert-base-uncased"
    dim: int = 128

    # Tokenizer
    qry_maxlen: int = 32
    doc_maxlen: int = 256

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