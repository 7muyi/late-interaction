import json
import os
from dataclasses import dataclass

import torch

from encoder import StrideTensor


@dataclass
class Encodings:
    mv_repr: StrideTensor
    cls_repr: torch.Tensor | None = None

    def lookup(self, start: int, num: int = 1) -> dict[str, torch.Tensor]:
        result = {"mv_repr": self.mv_repr.lookup(start, num)}
        if self.cls_repr is not None:
            result["cls_repr"] = self.cls_repr[start: start + num]
        return result


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file where each line is {"id": str, "text": str}.

    Returns a list of dicts in file order.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_encoding(encoding: dict, output_path: str) -> None:
    """Save each tensor in encoding as a .pt file under output_path/."""
    os.makedirs(output_path, exist_ok=True)
    for k, v in encoding.items():
        torch.save(v, os.path.join(output_path, f"{k}.pt"))


def load_encoding(dir_path: str, device: str) -> Encodings:
    """Load mv_repr.pt + mv_lens.pt (required) and cls_repr.pt (optional)."""
    mv_repr_path = os.path.join(dir_path, "mv_repr.pt")
    mv_lens_path = os.path.join(dir_path, "mv_lens.pt")
    cls_repr_path = os.path.join(dir_path, "cls_repr.pt")

    if not (os.path.exists(mv_repr_path) and os.path.exists(mv_lens_path)):
        raise FileNotFoundError(f"Missing mv_repr.pt or mv_lens.pt in {dir_path}")

    encoding = {
        "mv_repr": StrideTensor.from_packed_tensor(
            torch.load(mv_repr_path, weights_only=False),
            torch.load(mv_lens_path, weights_only=False),
            device
        )
    }

    if os.path.exists(cls_repr_path):
        encoding["cls_repr"] = torch.load(cls_repr_path, weights_only=False).to(device)

    return Encodings(**encoding)


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """Read TREC qrels: 'qid 0 did relevance' per line.

    Returns {qid_str: {did_str: relevance_int}}.
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            qid, did, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_results(path: str) -> dict[str, dict[str, float]]:
    """Read JSON results file written by retrieve().

    Coerces integer qid keys to strings for pytrec_eval compatibility.
    Returns {qid_str: {did_str: score_float}}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(qid): scores for qid, scores in data.items()}
