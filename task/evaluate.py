import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import argparse

from omegaconf import OmegaConf

from retriever.utils import load_qrels, load_results
from retriever.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config_path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args


def print_metrics(metrics: dict[str, float]) -> None:
    col_width = max(len(k) for k in metrics) + 2
    separator = "-" * (col_width + 10)

    print(separator)
    print(f"{'Metric':<{col_width}}{'Score':>8}")
    print(separator)

    for group in ("NDCG", "Recall", "MRR"):
        for key, val in metrics.items():
            if key.startswith(group):
                print(f"{key:<{col_width}}{val:>8.4f}")

    print(separator)


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    qrels = load_qrels(config.run.qrels_path)
    results = load_results(config.run.results_path)

    k_values = list(config.run.get("k_values", [1, 5, 10]))
    metrics = evaluate(results, qrels, k_values)

    print_metrics(metrics)
