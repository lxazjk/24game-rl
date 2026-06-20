from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .game24 import is_solvable
from .prompts import build_prompt


number_pattern = re.compile(r"\d+")


def parse_numbers(record: dict[str, Any], allowed_lengths: tuple[int, ...] = (4,)) -> list[int]:
    for key in ("numbers", "nums", "cards", "input", "Puzzles", "puzzles", "puzzle"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            numbers = [int(item) for item in value]
        else:
            numbers = [int(item) for item in number_pattern.findall(str(value))]
        if len(numbers) in allowed_lengths:
            return numbers
    expected = " or ".join(str(length) for length in allowed_lengths)
    raise ValueError(f"could not parse {expected} numbers from record keys: {sorted(record)}")


def parse_target(record: dict[str, Any], default: int = 24) -> int:
    for key in ("target", "Target", "goal", "Goal"):
        value = record.get(key)
        if value is not None:
            return int(value)
    return int(default)


def to_training_record(
    numbers: list[int],
    source: str,
    solvable: bool | None = None,
    target: int = 24,
) -> dict[str, Any]:
    return {
        "prompt": build_prompt(numbers, target=target),
        "numbers": numbers,
        "target": int(target),
        "source": source,
        "solvable": solvable,
    }


def parse_solved_rate(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            return float(stripped[:-1]) / 100.0
        return float(stripped)
    return float(value)


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_nlile_24_game() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset("nlile/24-game", split="train")
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for record in dataset:
        numbers = parse_numbers(record)
        solvable = parse_bool(record.get("solvable"), default=True)
        row = to_training_record(numbers, "nlile/24-game", solvable=solvable)
        if solvable:
            positives.append(row)
        else:
            negatives.append(row)
    return positives, negatives


def load_game_of_24() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("test-time-compute/game-of-24", split="train")
    rows: list[dict[str, Any]] = []
    for record in dataset:
        numbers = parse_numbers(record)
        row = to_training_record(
            numbers,
            "test-time-compute/game-of-24",
            solvable=is_solvable(numbers),
        )
        row["rank"] = record.get("Rank")
        row["solved_rate"] = parse_solved_rate(record.get("Solved rate"))
        rows.append(row)
    return rows


def load_countdown_3to4(
    train_limit: int,
    eval_limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.shuffle(seed=seed)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    seen_eval_keys: set[tuple[int, tuple[int, ...]]] = set()
    seen_train_keys: set[tuple[int, tuple[int, ...]]] = set()

    for index, record in enumerate(dataset):
        numbers = parse_numbers(record, allowed_lengths=(3, 4))
        target = parse_target(record)
        if any(number <= 0 for number in numbers):
            continue
        if not is_solvable(numbers, target=target):
            continue
        key = (target, tuple(sorted(numbers)))
        row = to_training_record(
            numbers,
            "Jiayi-Pan/Countdown-Tasks-3to4",
            solvable=True,
            target=target,
        )
        row["num_count"] = len(numbers)
        row["original_index"] = index

        if len(eval_rows) < eval_limit and key not in seen_eval_keys:
            eval_rows.append(row)
            seen_eval_keys.add(key)
            continue
        if len(train_rows) < train_limit and key not in seen_eval_keys and key not in seen_train_keys:
            train_rows.append(row)
            seen_train_keys.add(key)
        if len(train_rows) >= train_limit and len(eval_rows) >= eval_limit:
            break

    return train_rows, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare arithmetic-task JSONL files for GRPO.")
    parser.add_argument("--dataset", choices=("game24", "countdown"), default="game24")
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--eval-out", default="data/eval.jsonl")
    parser.add_argument("--negative-out", default="data/negative.jsonl")
    parser.add_argument("--limit-train-rows", type=int, default=5000)
    parser.add_argument("--limit-eval-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "countdown":
        train_rows, eval_rows = load_countdown_3to4(args.limit_train_rows, args.limit_eval_rows, args.seed)
        write_jsonl(args.train_out, train_rows)
        write_jsonl(args.eval_out, eval_rows)
        write_jsonl(args.negative_out, [])
        print(f"wrote {len(train_rows)} countdown train rows to {args.train_out}")
        print(f"wrote {len(eval_rows)} countdown eval rows to {args.eval_out}")
        print(f"wrote 0 negative rows to {args.negative_out}")
        return

    positives, negatives = load_nlile_24_game()
    eval_rows = load_game_of_24()
    train_keys = {tuple(row["numbers"]) for row in positives}
    for row in eval_rows:
        row["seen_in_train"] = tuple(row["numbers"]) in train_keys

    hard_eval = [
        row
        for row in eval_rows
        if row["solvable"] and row.get("solved_rate") is not None
    ]
    hard_eval.sort(key=lambda row: row["solved_rate"])
    hard_eval = hard_eval[:100] if len(hard_eval) >= 100 else hard_eval

    write_jsonl(args.train_out, positives)
    write_jsonl(args.eval_out, hard_eval)
    write_jsonl(args.negative_out, negatives)

    print(f"wrote {len(positives)} train rows to {args.train_out}")
    print(f"wrote {len(hard_eval)} hard eval rows to {args.eval_out}")
    print(f"wrote {len(negatives)} negative rows to {args.negative_out}")


if __name__ == "__main__":
    main()
