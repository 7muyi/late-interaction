from dataclasses import dataclass


@dataclass
class Config:
    # TokenPooling parameters
    pretrained_model: str = "bert-base-uncased"
    dim: int = 128
    pooling_factor: int = 2  # Evaluated values: 2, 3, 4, 5, 6, 8. (e.g., 2 for 50% reduction)

    # Tokenzier
    qry_maxlen: int = 32
    doc_maxlen: int = 256

    # Training
    dataset_path: str = ""
    checkpoint_path: str = ""

    amp: bool = True
    epoch: int = 5
    bsize: int = 128
    # accumulation_steps: int = 2
    # warmup: int = 1000
    # lr_backbone: float = 3e-6
    # lr_other: float = 1e-5
    # lr_min_ratio: float = 0.1

    log_interval: int = 25

    temperature: float = 0.1