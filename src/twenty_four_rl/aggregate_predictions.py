from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .evaluate import score_completion, write_jsonl


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def puzzle_key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(number) for number in row["numbers"])


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, ...], dict[str, Any]]:
    counts = Counter(puzzle_key(row) for row in rows)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate puzzle keys are ambiguous: {duplicates[:5]}")
    return {puzzle_key(row): row for row in rows}


def attempts_from_prediction(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "attempts" in row and row["attempts"]:
        return list(row["attempts"])
    return [
        {
            "sample_index": 0,
            "completion": row.get("completion", ""),
        }
    ]


def collect_attempts(
    prediction_paths: list[str | Path],
) -> tuple[dict[tuple[int, ...], list[dict[str, Any]]], dict[str, Any]]:
    attempts_by_key: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    source_budgets: dict[str, dict[str, int]] = {}

    for path in prediction_paths:
        source = str(path)
        rows = read_jsonl(path)
        source_candidate_count = 0
        for row_index, row in enumerate(rows):
            key = puzzle_key(row)
            row_attempts = attempts_from_prediction(row)
            source_candidate_count += len(row_attempts)
            for attempt in row_attempts:
                attempts_by_key[key].append(
                    {
                        "completion": attempt.get("completion", ""),
                        "source_file": source,
                        "source_row_index": row_index,
                        "source_sample_index": attempt.get("sample_index", 0),
                    }
                )
        source_budgets[source] = {
            "rows": len(rows),
            "candidates": source_candidate_count,
        }
    return attempts_by_key, source_budgets


def aggregate_predictions(
    input_rows: list[dict[str, Any]],
    prediction_paths: list[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_by_key = index_rows(input_rows)
    attempts_by_key, source_budgets = collect_attempts(prediction_paths)
    aggregated: list[dict[str, Any]] = []

    for key, row in input_by_key.items():
        raw_attempts = attempts_by_key.get(key, [])
        scored_attempts: list[dict[str, Any]] = []
        for global_index, raw_attempt in enumerate(raw_attempts):
            scored = score_completion(raw_attempt["completion"], row["numbers"], global_index)
            scored.update(
                {
                    "source_file": raw_attempt["source_file"],
                    "source_row_index": raw_attempt["source_row_index"],
                    "source_sample_index": raw_attempt["source_sample_index"],
                }
            )
            scored_attempts.append(scored)

        valid_attempts = [attempt for attempt in scored_attempts if attempt["valid"]]
        best_attempt = valid_attempts[0] if valid_attempts else (scored_attempts[0] if scored_attempts else None)
        first_valid_sample_index = best_attempt["sample_index"] if best_attempt and best_attempt["valid"] else None
        aggregated_row = {
            **row,
            "num_samples": len(scored_attempts),
            "valid": bool(best_attempt and best_attempt["valid"]),
            "completion": best_attempt["completion"] if best_attempt else "",
            "value": best_attempt["value"] if best_attempt else None,
            "reason": best_attempt["reason"] if best_attempt else "no candidates",
            "expression": best_attempt["expression"] if best_attempt else "",
            "first_valid_sample_index": first_valid_sample_index,
            "candidates_until_solution": first_valid_sample_index + 1 if first_valid_sample_index is not None else None,
            "num_valid_attempts": len(valid_attempts),
            "valid_attempt_rate": len(valid_attempts) / len(scored_attempts) if scored_attempts else 0.0,
            "attempts": scored_attempts,
        }
        aggregated.append(aggregated_row)

    solved = sum(bool(row["valid"]) for row in aggregated)
    candidate_counts = [int(row["num_samples"]) for row in aggregated]
    candidates_until_solution = [
        int(row["candidates_until_solution"])
        for row in aggregated
        if row["candidates_until_solution"] is not None
    ]
    total_candidates = sum(candidate_counts)
    summary: dict[str, Any] = {
        "count": len(aggregated),
        "solved": solved,
        "solved_rate": solved / len(aggregated) if aggregated else None,
        "total_model_candidates": total_candidates,
        "mean_candidate_budget_per_problem": mean(candidate_counts) if candidate_counts else None,
        "max_candidate_budget_per_problem": max(candidate_counts) if candidate_counts else None,
        "min_candidate_budget_per_problem": min(candidate_counts) if candidate_counts else None,
        "mean_candidates_until_solution": mean(candidates_until_solution) if candidates_until_solution else None,
        "median_candidates_until_solution": median(candidates_until_solution) if candidates_until_solution else None,
        "max_candidates_until_solution": max(candidates_until_solution) if candidates_until_solution else None,
        "source_budgets": source_budgets,
        "unsolved_keys": [list(puzzle_key(row)) for row in aggregated if not row["valid"]],
        "reason_counts": dict(
            Counter(
                attempt["reason"] or "ok"
                for row in aggregated
                for attempt in row["attempts"]
            ).most_common()
        ),
    }
    return aggregated, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate verifier-selected 24-game prediction files.")
    parser.add_argument("--input", required=True, help="Original JSONL with prompt and numbers fields.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction JSONL to include. May be passed multiple times.",
    )
    parser.add_argument("--output", help="Optional aggregated prediction JSONL output.")
    parser.add_argument("--summary-output", help="Optional JSON summary output.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    aggregated, summary = aggregate_predictions(rows, args.predictions)
    print(json.dumps(summary, ensure_ascii=False))
    if args.output:
        write_jsonl(args.output, aggregated)
    if args.summary_output:
        write_json(args.summary_output, summary)


if __name__ == "__main__":
    main()
