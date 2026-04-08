import argparse


def parse_dict(arg: str) -> dict:
    try:
        return dict(pair.split("=") for pair in arg.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Parameters must be in key=value format and separated by commas.")

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="colbert")
    parser.add_argument("--kwargs", type=parse_dict)
    return parser