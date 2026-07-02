from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build import parse_numbers
from core.game24 import validate_expression
from core.prompts import build_prompt


def normalize_solution(solution: str) -> str:
    return (
        solution.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace(" ", "")
        .strip()
    )


def choose_solution(record: dict[str, Any]) -> str | None:
    numbers = parse_numbers(record)
    for raw_solution in record.get("solutions", []):
        expression = normalize_solution(str(raw_solution))
        if validate_expression(expression, numbers).ok:
            return expression
    return None


def make_text(numbers: list[int], expression: str) -> str:
    prompt = build_prompt(numbers)
    completion = (
        "<think>Combine the four numbers with arithmetic and verify the result is 24.</think>"
        f"<answer>{expression}</answer>"
    )
    return prompt + "\n" + completion


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT warmup data from nlile/24-game solutions.")
    parser.add_argument("--train-out", default="data/sft_train.jsonl")
    parser.add_argument("--eval-out", default="data/sft_eval.jsonl")
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("nlile/24-game", split="train")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in dataset:
        numbers = parse_numbers(record)
        expression = choose_solution(record)
        if expression is None:
            skipped += 1
            continue
        rows.append(
            {
                "text": make_text(numbers, expression),
                "numbers": numbers,
                "expression": expression,
                "source": "nlile/24-game",
            }
        )

    eval_rows = rows[-args.eval_size :] if args.eval_size else []
    train_rows = rows[: -args.eval_size] if args.eval_size else rows
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    print(f"wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"wrote {len(eval_rows)} eval rows to {args.eval_out}")
    print(f"skipped {skipped} rows without a validated solution")


if __name__ == "__main__":
    main()
