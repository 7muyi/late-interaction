import sys
sys.path.append("/root/late-interaction")

from colbert.config import Config
from colbert.colbert import ColBERT
from colbert.tokenizer import Tokenizer

from train import run


if __name__ == "__main__":
    config = Config()
    model = ColBERT(config.pretrained_model, config.dim)
    tokenizer = Tokenizer(config.pretrained_model, config.qry_maxlen, config.doc_maxlen)

    run(model, tokenizer, config)
