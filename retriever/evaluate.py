import pytrec_eval  # type: ignore


def evaluate(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute NDCG@k, Recall@k, and MRR over all queries.

    Args:
        results: {qid_str: {did_str: score}}, as returned by retrieve().
        qrels:   {qid_str: {did_str: relevance_int}}, as returned by load_qrels().
        k_values: list of cutoffs, e.g. [1, 5, 10].

    Returns:
        Flat dict of averaged metrics, e.g.
        {'NDCG@10': 0.65, 'Recall@10': 0.72, 'MRR': 0.61, ...}.
    """
    metrics = {f"ndcg_cut_{k}" for k in k_values} | {f"recall_{k}" for k in k_values}
    metrics.add("recip_rank")  # MRR is cutoff-independent

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query = evaluator.evaluate(results)

    output = {}
    for k in k_values:
        output[f"NDCG@{k}"] = _mean(per_query, f"ndcg_cut_{k}")
        output[f"Recall@{k}"] = _mean(per_query, f"recall_{k}")
    output["MRR"] = _mean(per_query, "recip_rank")
    return output


def _mean(per_query: dict, metric: str) -> float:
    vals = [v[metric] for v in per_query.values() if metric in v]
    return sum(vals) / len(vals) if vals else 0.0
