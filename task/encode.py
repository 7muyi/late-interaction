import os
import pathlib
import sys
sys.path.append(pathlib.Path(__file__).parent.parent.absolute())

import torch

from encoder import Encoder
from models import model_factory
from utils import load_checkpoint
from .argparser import get_argparser


def load_texts(path: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.split("\t")[1].strip()  # format: "id text"
            texts.append(text)
    return texts


def save(repr: dict, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    for k, v in repr.items():
        torch.save(v, os.path.join(output_path, f"{k}.pt"))


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    kwargs = args.kwargs

    ckpt = load_checkpoint(path=kwargs["ckpt_path"])
    model, tokenizer, _ = model_factory(args.model, **ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])

    encoder = Encoder(model, tokenizer, device="cuda")

    texts = load_texts(kwargs["data_path"])

    file_name = os.path.basename(kwargs["data_path"])
    if file_name == "collections.tsv":
        encoding = encoder.encode_doc(texts, int(kwargs["bsize"]))
    elif file_name == "queries.tsv":
        encoding = encoder.encode_qry(texts, int(kwargs["bsize"]))
    else:
        raise ValueError("Input file must be either collections.tsv or queries.tsv")

    save(encoding, kwargs["output_path"])