import argparse
from omegaconf import OmegaConf

from common import registry
from train.utils import setup_ddp, cleanup_ddp, set_seed
from train.runner import run
from dataset import get_dataloader

# Import to trigger @registry.register_* decorators
import models.colbert, models.constbert, models.msbert, models.tokenpooling  # noqa: F401
import tokenizers.std_tokenizer, tokenizers.span_tokenizer  # noqa: F401
import train.optim  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config_path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    config = OmegaConf.load(args.config_path)
    setup_ddp(config.run)

    set_seed(config.run.local_rank + 42)

    model = registry.get_model_cls(config.model.name).from_config(config.model)
    tokenizer = registry.get_tokenizer_cls(config.tokenizer.name).from_config(config.tokenizer)
    dataloader = get_dataloader(
        config.run.triplet_path,
        config.run.queries_path,
        config.run.corpus_path,
        tokenizer,
        config.run.bsize,
        config.run.rank,
        config.run.world_size
    )

    run(model, dataloader, config.run)

    cleanup_ddp()