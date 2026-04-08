from dataclasses import asdict, dataclass


@dataclass
class BaseConfig:
    def to_dict(self) -> dict:
        return asdict(self)