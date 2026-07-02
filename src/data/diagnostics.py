from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def number_key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(int(number) for number in row["numbers"]))


def summarize_data(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_keys = {number_key(row) for row in train_rows}
    eval_keys = {number_key(row) for row in eval_rows}
    eval_seen_flags = [bool(row.get("seen_in_train")) for row in eval_rows]
    eval_overlap_keys = eval_keys & train_keys
    eval_overlap_rows = [row for row in eval_rows if number_key(row) in train_keys]

    return {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "negative_count": len(negative_rows),
        "negative_empty": len(negative_rows) == 0,
        "train_unique_number_sets": len(train_keys),
        "eval_unique_number_sets": len(eval_keys),
        "eval_seen_in_train_count": sum(eval_seen_flags),
        "eval_seen_in_train_rate": sum(eval_seen_flags) / len(eval_rows) if eval_rows else None,
        "eval_overlap_unique_count": len(eval_overlap_keys),
        "eval_overlap_unique_rate": len(eval_overlap_keys) / len(eval_keys) if eval_keys else None,
        "eval_overlap_row_count": len(eval_overlap_rows),
        "eval_overlap_row_rate": len(eval_overlap_rows) / len(eval_rows) if eval_rows else None,
        "eval_solvable_count": sum(bool(row.get("solvable", True)) for row in eval_rows),
        "eval_solvable_rate": (
            sum(bool(row.get("solvable", True)) for row in eval_rows) / len(eval_rows)
            if eval_rows
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 24-game data counts and train/eval overlap.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    summary = summarize_data(
        read_jsonl(args.train),
        read_jsonl(args.eval),
        read_jsonl(args.negative),
    )
    print(json.dumps(summary, ensure_ascii=False))
    if args.summary_output:
        write_json(args.summary_output, summary)


if __name__ == "__main__":
    main()
