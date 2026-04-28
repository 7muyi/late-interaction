import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import argparse

from omegaconf import OmegaConf

from encoder import Encoder
from common import registry

# Import to trigger @registry.register_* decorators
from models import *
from tokenizer import *

from retriever.utils import load_jsonl
from retriever.encode import encode_texts


def parse_args():
    parser = argparse.ArgumentParser(description="Encoding")
    parser.add_argument("--config_path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    model = registry.get_model_cls(config.model.name).from_pretrained(args.config_path)
    tokenizer = registry.get_tokenizer_cls(config.tokenizer.name).from_config(config.tokenizer)

    encoder = Encoder(model, tokenizer, device="cpu")

    records = load_jsonl(config.run.dataset_path)
    texts = [r["text"] for r in records]

    file_name = os.path.basename(config.run.dataset_path)
    if file_name == "queries.jsonl":
        target = "qry"
    elif file_name == "corpus.jsonl":
        target = "doc"
    else:
        raise ValueError("Input file must be either queries.jsonl or corpus.jsonl")

    encode_texts(
        encoder,
        texts,
        target=target,
        devices=config.run.devices,
        bsize=int(config.run.get("bsize", None)),
        output_path=config.run.get("output_path", None)
    )
