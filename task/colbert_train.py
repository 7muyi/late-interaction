import sys
sys.path.append("/root/late-interaction")

from models import model_factory

from train import run


if __name__ == "__main__":
    model, tokenizer, config = model_factory("colbert")

    run(model, tokenizer, config)
