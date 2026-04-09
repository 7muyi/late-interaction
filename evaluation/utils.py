from dataclasses import dataclass
import os

import torch

from encoder import StrideTensor
from models import _load_model_components



@dataclass
class Encodings:
    mv_repr: StrideTensor
    cls_repr: torch.Tensor | None = None

    def lookup(self, start: int, num: int = 1) -> dict[str, torch.Tensor]:
        result = {"mv_repr": self.mv_repr.lookup(start, num)}
        if self.cls_repr is not None:
            result["cls_repr"] = self.cls_repr[start: start + num]
        return result


def load_encoding(dir_path: str, device: str) -> dict[str, torch.Tensor | StrideTensor]:
    mv_repr_path = os.path.join(dir_path, "mv_repr.pt")
    mv_lens_path = os.path.join(dir_path, "mv_lens.pt")
    cls_repr_path = os.path.join(dir_path, "cls_repr.pt")

    encoding = {}

    if os.path.exists(mv_repr_path) and os.path.exists(mv_lens_path):
        encoding["mv_repr"] = StrideTensor.from_packed_tensor(
            torch.load(mv_repr_path, weights_only=False),
            torch.load(mv_lens_path, weights_only=False),
            device
        )
    else:
        raise FileNotFoundError(f"Missing mv_repr.pt or mv_lens.pt in {dir_path}")

    if os.path.exists(cls_repr_path):
        encoding["cls_repr"] = torch.load(cls_repr_path, weights_only=False).to(device)

    return Encodings(**encoding)


def get_score_func(model_name: str):
    _, ModelCls, _ = _load_model_components(model_name)
    return ModelCls.score