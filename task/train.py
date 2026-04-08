import pathlib
import sys
sys.path.append(pathlib.Path(__file__).parent.parent.absolute())

from models import model_factory

from train import run
from .argparser import get_argparser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    model, tokenizer, config = model_factory(args.model, **args.kwargs)
    run(model, tokenizer, config)