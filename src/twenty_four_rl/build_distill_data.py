from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prompts import build_prompt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_verified_text(numbers: list[int], completion: str, expression: str) -> str:
    prompt = build_prompt(numbers)
    if "<think" in completion.lower() and "<answer" in completion.lower():
        return prompt + "\n" + completion.strip()
    return (
        prompt
        + "\n"
        + "<think>Use the four numbers exactly once and verify the expression equals 24.</think>"
        + f"<answer>{expression}</answer>"
    )


def build_distill_rows(
    prediction_rows: list[dict[str, Any]],
    max_solutions_per_puzzle: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    puzzles_with_solution = 0
    total_valid_attempts = 0

    for prediction_row in prediction_rows:
        numbers = [int(number) for number in prediction_row["numbers"]]
        attempts = prediction_row.get("attempts") or [prediction_row]
        seen_expressions: set[str] = set()
        kept_for_puzzle = 0

        for attempt in attempts:
            if not attempt.get("valid"):
                continue
            expression = str(attempt.get("expression") or "").strip()
            completion = str(attempt.get("completion") or "").strip()
            if not expression or expression in seen_expressions:
                continue
            seen_expressions.add(expression)
            total_valid_attempts += 1
            rows.append(
                {
                    "text": make_verified_text(numbers, completion, expression),
                    "numbers": numbers,
                    "expression": expression,
                    "source": "verified_sample",
                    "sample_index": attempt.get("sample_index"),
                }
            )
            kept_for_puzzle += 1
            if kept_for_puzzle >= max_solutions_per_puzzle:
                break

        if kept_for_puzzle:
            puzzles_with_solution += 1

    summary = {
        "input_rows": len(prediction_rows),
        "output_rows": len(rows),
        "puzzles_with_solution": puzzles_with_solution,
        "total_valid_attempts_seen_until_caps": total_valid_attempts,
        "max_solutions_per_puzzle": max_solutions_per_puzzle,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT data from verifier-approved model samples.")
    parser.add_argument("--input", required=True, help="Prediction JSONL with attempts from evaluate.py")
    parser.add_argument("--output", required=True, help="Output SFT JSONL")
    parser.add_argument("--summary-output", help="Optional summary JSON")
    parser.add_argument("--max-solutions-per-puzzle", type=int, default=1)
    args = parser.parse_args()

    if args.max_solutions_per_puzzle < 1:
        parser.error("--max-solutions-per-puzzle must be >= 1")

    rows, summary = build_distill_rows(read_jsonl(args.input), args.max_solutions_per_puzzle)
    write_jsonl(args.output, rows)
    print(json.dumps(summary, ensure_ascii=False))
    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
