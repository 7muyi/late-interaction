import os
import json

import torch
from tqdm import tqdm

from common import registry
import scorer  # noqa: F401 — side-effect: registers scorer functions

from retriever.utils import Encodings


def retrieve(
    Q: Encodings,
    D: Encodings,
    score_func_name: str,
    qry_bsize: int,
    doc_bsize: int,
    show_progress: bool = True,
    topk: int = 10,
    output_path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Score all query-document pairs and return top-k per query.

    Returns {qid_str: {did_str: score}} for pytrec_eval compatibility.
    If output_path is provided, saves the results as a JSON file.
    """
    score_func = registry.get_scorer(score_func_name)

    results = {}
    qid = 0
    for i in tqdm(range(0, len(Q.mv_repr), qry_bsize), disable=not show_progress):
        Q_batch = Q.lookup(i, qry_bsize)

        all_scores = []
        for j in tqdm(range(0, len(D.mv_repr), doc_bsize), disable=True):
            D_batch = D.lookup(j, doc_bsize)
            scores = score_func(Q_batch, D_batch, pairwise=False)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, -1)
        k = min(topk, all_scores.shape[-1])
        topk_vals, topk_indices = torch.topk(all_scores, k=k, dim=-1)

        for qi in range(topk_indices.shape[0]):
            results[str(qid)] = {
                str(idx.item()): float(val)
                for idx, val in zip(topk_indices[qi], topk_vals[qi])
            }
            qid += 1

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f"result_{topk}.json"), "w") as f:
            json.dump(results, f, indent=4)

    return results
