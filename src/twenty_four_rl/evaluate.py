from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .game24 import solve_24, validate_expression


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


def score_completion(completion: str, numbers: list[int], sample_index: int) -> dict[str, Any]:
    result = validate_expression(completion, numbers)
    return {
        "sample_index": sample_index,
        "completion": completion,
        "valid": result.ok,
        "value": float(result.value) if result.value is not None else None,
        "reason": result.reason,
        "expression": result.expression,
    }


def evaluate_completions(
    rows: list[dict[str, Any]],
    completions_by_row: list[list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated = []
    solved = 0

    for row, completions in zip(rows, completions_by_row):
        if not completions:
            raise ValueError("empty completion list")

        attempts = [
            score_completion(completion, row["numbers"], sample_index)
            for sample_index, completion in enumerate(completions)
        ]
        best_attempt = next((attempt for attempt in attempts if attempt["valid"]), attempts[0])
        solved += int(best_attempt["valid"])
        evaluated_row = {
            **row,
            "num_samples": len(completions),
            "completion": best_attempt["completion"],
            "valid": best_attempt["valid"],
            "value": best_attempt["value"],
            "reason": best_attempt["reason"],
            "expression": best_attempt["expression"],
        }
        if len(completions) > 1:
            evaluated_row["attempts"] = attempts
        evaluated.append(evaluated_row)

    solved_rate = solved / len(rows) if rows else 0.0
    summary = {"count": len(rows), "solved": solved, "solved_rate": solved_rate}
    return evaluated, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 24-game responses.")
    parser.add_argument("--input", required=True, help="JSONL with prompt and numbers fields.")
    parser.add_argument("--model", help="Model/checkpoint path. Omit with --oracle-solver.")
    parser.add_argument("--output", help="Optional JSONL prediction output.")
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
            expression = solve_24(row["numbers"])
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


if __name__ == "__main__":
    main()
