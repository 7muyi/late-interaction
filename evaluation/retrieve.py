import torch
from tqdm import tqdm

from .utils import load_encoding, get_score_func


def retrieve(
    model_name: str,
    qry_encoding_dir: str,
    doc_encoding_dir: str,
    qry_bsize: int,
    doc_bsize: int,
    device: str,
    show_progress: bool = True,
    topk: int = 10,
    output_path: str | None = None
) -> None:
    Q = load_encoding(qry_encoding_dir, device)
    D = load_encoding(doc_encoding_dir, device)

    score_func = get_score_func(model_name)

    results = {}
    qid = 0
    for i in tqdm(range(0, len(Q.mv_repr), qry_bsize), disable=not show_progress):
        Q_batch = Q.lookup(i, qry_bsize)

        all_scores = []
        for j in tqdm(range(0, len(D.mv_repr), doc_bsize), disable=not show_progress):
            D_batch = D.lookup(j, doc_bsize)

            scores = score_func(Q_batch, D_batch, pairwise=False)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, -1)
        topk_vals, topk_indices = torch.topk(all_scores, k=topk, dim=-1)

        for k in range(topk_indices.shape[0]):
            results[qid] = {str(idx.item()): float(val) for idx, val in zip(topk_indices[k], topk_vals[k])}
            qid += 1

    if output_path is not None:
        with open(output_path, "w") as f:
            import json
            json.dump(results, f, indent=4)

    return results
