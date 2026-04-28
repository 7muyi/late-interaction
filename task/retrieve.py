import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import time

from omegaconf import OmegaConf

from retriever.utils import load_encoding
from retriever.retrieve import retrieve


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval")
    parser.add_argument("--config_path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    # Loading
    t0 = time.perf_counter()
    Q = load_encoding(config.run.qry_encoding_dir, config.run.device)
    t_qry = time.perf_counter() - t0

    t0 = time.perf_counter()
    D = load_encoding(config.run.doc_encoding_dir, config.run.device)
    t_doc = time.perf_counter() - t0

    print("[Loading]")
    print(f"  Query encoding : {t_qry:.2f} s")
    print(f"  Doc encoding   : {t_doc:.2f} s")

    # Retrieval
    t0 = time.perf_counter()
    results = retrieve(
        Q=Q,
        D=D,
        score_func_name=config.run.score_func_name,
        qry_bsize=int(config.run.qry_bsize),
        doc_bsize=int(config.run.doc_bsize),
        show_progress=config.run.get("show_progress", True),
        topk=int(config.run.get("topk", 10)),
        output_path=config.run.get("output_path", None),
    )
    t_score = time.perf_counter() - t0

    n = len(Q.mv_repr)
    print("\n[Retrieval]")
    print(f"  Queries        : {n}")
    print(f"  Scoring time   : {t_score:.2f} s")
    print(f"  Latency/query  : {t_score / n * 1000:.2f} ms")
    print(f"  Throughput     : {n / t_score:.1f} queries/s")