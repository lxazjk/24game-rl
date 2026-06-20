from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from .game24 import analyze_expression, solve_24


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_with_model(
    model_name_or_path: str,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_samples: int,
) -> list[list[str]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outputs: list[list[str]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = generator(
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=num_samples,
            return_full_text=False,
        )
        for result in results:
            outputs.append([item["generated_text"] for item in result])
    return outputs


def score_completion(completion: str, numbers: list[int], sample_index: int, target: int = 24) -> dict[str, Any]:
    analysis = analyze_expression(completion, numbers, target=target)
    return {
        "sample_index": sample_index,
        "completion": completion,
        "valid": analysis.valid,
        "has_r1_format": analysis.has_r1_format,
        "has_answer": analysis.has_answer,
        "allowed_chars": analysis.allowed_chars,
        "parse_ok": analysis.parse_ok,
        "numbers_ok": analysis.numbers_ok,
        "target_ok": analysis.target_ok,
        "used_numbers": list(analysis.used_numbers),
        "value": float(analysis.value) if analysis.value is not None else None,
        "reason": analysis.reason,
        "expression": analysis.expression,
    }


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _diagnostic_rates(prefix: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(attempts)
    return {
        f"{prefix}_count": total,
        f"{prefix}_r1_format_rate": _rate(sum(bool(a["has_r1_format"]) for a in attempts), total),
        f"{prefix}_answer_tag_rate": _rate(sum(bool(a["has_answer"]) for a in attempts), total),
        f"{prefix}_allowed_chars_rate": _rate(sum(bool(a["allowed_chars"]) for a in attempts), total),
        f"{prefix}_parse_ok_rate": _rate(sum(bool(a["parse_ok"]) for a in attempts), total),
        f"{prefix}_numbers_ok_rate": _rate(sum(bool(a["numbers_ok"]) for a in attempts), total),
        f"{prefix}_target_ok_rate": _rate(sum(bool(a["target_ok"]) for a in attempts), total),
        f"{prefix}_valid_rate": _rate(sum(bool(a["valid"]) for a in attempts), total),
    }


def summarize_evaluated_rows(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(evaluated)
    if count == 0:
        return {
            "count": 0,
            "solved": 0,
            "solved_rate": None,
            "pass_at_1_count": 0,
            "pass_at_1_rate": None,
            "solved_at_n_count": 0,
            "solved_at_n_rate": None,
            "empty": True,
        }

    first_attempts = [row["attempts"][0] for row in evaluated]
    all_attempts = [attempt for row in evaluated for attempt in row["attempts"]]
    solved_count = sum(bool(row["valid"]) for row in evaluated)
    pass_at_1_count = sum(bool(attempt["valid"]) for attempt in first_attempts)
    solved_rows = [row for row in evaluated if row["candidates_until_solution"] is not None]
    candidates_until_solution = [int(row["candidates_until_solution"]) for row in solved_rows]
    max_samples = max(int(row["num_samples"]) for row in evaluated)

    summary: dict[str, Any] = {
        "count": count,
        "solved": solved_count,
        "solved_rate": _rate(solved_count, count),
        "solved_at_n_count": solved_count,
        "solved_at_n_rate": _rate(solved_count, count),
        "pass_at_1_count": pass_at_1_count,
        "pass_at_1_rate": _rate(pass_at_1_count, count),
        "candidate_count_total": sum(int(row["num_samples"]) for row in evaluated),
        "candidate_count_mean_per_problem": mean(int(row["num_samples"]) for row in evaluated),
        "num_samples_max": max_samples,
        "num_samples_min": min(int(row["num_samples"]) for row in evaluated),
        "reason_counts": dict(Counter(attempt["reason"] or "ok" for attempt in all_attempts).most_common()),
        "empty": False,
    }
    summary.update(_diagnostic_rates("first_attempt", first_attempts))
    summary.update(_diagnostic_rates("attempt", all_attempts))

    if candidates_until_solution:
        summary.update(
            {
                "mean_candidates_until_solution": mean(candidates_until_solution),
                "median_candidates_until_solution": median(candidates_until_solution),
                "max_candidates_until_solution": max(candidates_until_solution),
            }
        )
    else:
        summary.update(
            {
                "mean_candidates_until_solution": None,
                "median_candidates_until_solution": None,
                "max_candidates_until_solution": None,
            }
        )

    for milestone in (1, 4, 16, 64, 128):
        if max_samples >= milestone:
            solved_by_milestone = 0
            for row in evaluated:
                attempts = row["attempts"][:milestone]
                solved_by_milestone += int(any(attempt["valid"] for attempt in attempts))
            summary[f"solved_at_{milestone}_count"] = solved_by_milestone
            summary[f"solved_at_{milestone}"] = _rate(solved_by_milestone, count)

    return summary


def evaluate_completions(
    rows: list[dict[str, Any]],
    completions_by_row: list[list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated = []

    if len(rows) != len(completions_by_row):
        raise ValueError("rows and completions_by_row must have the same length")

    for row, completions in zip(rows, completions_by_row):
        if not completions:
            raise ValueError("empty completion list")

        attempts = [
            score_completion(completion, row["numbers"], sample_index, target=int(row.get("target", 24)))
            for sample_index, completion in enumerate(completions)
        ]
        valid_attempts = [attempt for attempt in attempts if attempt["valid"]]
        best_attempt = valid_attempts[0] if valid_attempts else attempts[0]
        first_valid_sample_index = best_attempt["sample_index"] if best_attempt["valid"] else None
        evaluated_row = {
            **row,
            "num_samples": len(completions),
            "completion": best_attempt["completion"],
            "valid": best_attempt["valid"],
            "value": best_attempt["value"],
            "reason": best_attempt["reason"],
            "expression": best_attempt["expression"],
            "first_valid_sample_index": first_valid_sample_index,
            "candidates_until_solution": first_valid_sample_index + 1 if first_valid_sample_index is not None else None,
            "num_valid_attempts": len(valid_attempts),
            "valid_attempt_rate": len(valid_attempts) / len(attempts),
            "pass_at_1": attempts[0]["valid"],
            "attempts": attempts,
        }
        evaluated.append(evaluated_row)

    return evaluated, summarize_evaluated_rows(evaluated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 24-game responses.")
    parser.add_argument("--input", required=True, help="JSONL with prompt and numbers fields.")
    parser.add_argument("--model", help="Model/checkpoint path. Omit with --oracle-solver.")
    parser.add_argument("--output", help="Optional JSONL prediction output.")
    parser.add_argument("--summary-output", help="Optional JSON summary output.")
    parser.add_argument("--oracle-solver", action="store_true", help="Use exhaustive search instead of a model.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, help="Evaluate only the first N rows.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1, help="Number of completions per prompt.")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if args.num_samples > 1 and not args.do_sample:
        parser.error("--num-samples > 1 requires --do-sample")

    rows = read_jsonl(args.input)
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    if args.oracle_solver:
        completions_by_row = []
        for row in rows:
            target = int(row.get("target", 24))
            expression = solve_24(row["numbers"], target=target)
            completion = f"<think>exhaustive search</think><answer>{expression or ''}</answer>"
            completions_by_row.append([completion])
    else:
        if not args.model:
            raise SystemExit("--model is required unless --oracle-solver is set")
        completions_by_row = generate_with_model(
            args.model,
            [row["prompt"] for row in rows],
            args.max_new_tokens,
            args.batch_size,
            args.do_sample,
            args.temperature,
            args.top_p,
            args.num_samples,
        )

    evaluated, summary = evaluate_completions(rows, completions_by_row)
    summary["num_samples"] = args.num_samples
    summary[f"solved_at_{args.num_samples}"] = summary["solved_rate"]
    print(json.dumps(summary, ensure_ascii=False))
    if args.output:
        write_jsonl(args.output, evaluated)
    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
