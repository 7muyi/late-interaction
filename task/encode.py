import argparse
import os

from omegaconf import OmegaConf

from encode import Encoder
from common import registry

# Import to trigger @registry.register_* decorators
from models import *
from tokenizer import *

from retrieve.utils import load_jsonl
from retrieve.encode import encode_queries, encode_docs


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

    encoder = Encoder(model, tokenizer, device=config.run.device)

    records = load_jsonl(config.run.dataset_path)
    texts = [r["text"] for r in records]

    file_name = os.path.basename(config.run.dataset_path)
    if file_name == "queries.jsonl":
        encode_queries(encoder, texts, int(config.run.bsize), output_path=config.run.output_path)
    elif file_name == "collections.jsonl":
        encode_docs(encoder, texts, int(config.run.bsize), output_path=config.run.output_path)
    else:
        raise ValueError("Input file must be either queries.jsonl or collections.jsonl")
