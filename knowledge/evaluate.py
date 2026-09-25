import argparse
import json
from pathlib import Path

from knowledge.retriever import Retriever, create_default_retriever

DEFAULT_EVALUATION_PATH = Path(__file__).parent / "data" / "retrieval_eval_v1.json"


def load_evaluation_cases(path: Path = DEFAULT_EVALUATION_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as evaluation_file:
        payload = json.load(evaluation_file)
    if payload.get("version") != "1.0":
        raise ValueError("Unsupported retrieval-evaluation version")
    return payload["cases"]


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[dict],
    top_k: int,
) -> dict:
    """Calculate source-level Hit Rate@K and Recall@K.

    Hit Rate is the fraction of queries retrieving at least one expected document.
    Recall is the average fraction of each query's expected documents retrieved.
    """
    if not cases:
        raise ValueError("At least one evaluation case is required")

    hits = 0
    recalls = []
    case_results = []
    for case in cases:
        expected = set(case["expected_document_ids"])
        retrieved_chunks = retriever.retrieve(case["query"], top_k)
        retrieved = {chunk.document_id for chunk in retrieved_chunks}
        matched = expected.intersection(retrieved)
        hit = bool(matched)
        recall = len(matched) / len(expected)
        hits += int(hit)
        recalls.append(recall)
        case_results.append(
            {
                "query": case["query"],
                "expected_document_ids": sorted(expected),
                "retrieved_document_ids": sorted(retrieved),
                "hit": hit,
                "recall": round(recall, 4),
            }
        )

    return {
        "case_count": len(cases),
        "top_k": top_k,
        "hit_rate_at_k": round(hits / len(cases), 4),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "cases": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local retrieval index")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    metrics = evaluate_retrieval(
        create_default_retriever(),
        load_evaluation_cases(),
        args.top_k,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
