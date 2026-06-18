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


def generate_with_model(model_name_or_path: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outputs: list[str] = []
    for prompt in prompts:
        result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
        outputs.append(result[0]["generated_text"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 24-game responses.")
    parser.add_argument("--input", required=True, help="JSONL with prompt and numbers fields.")
    parser.add_argument("--model", help="Model/checkpoint path. Omit with --oracle-solver.")
    parser.add_argument("--output", help="Optional JSONL prediction output.")
    parser.add_argument("--oracle-solver", action="store_true", help="Use exhaustive search instead of a model.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.oracle_solver:
        completions = []
        for row in rows:
            expression = solve_24(row["numbers"])
            completions.append(f"<think>exhaustive search</think><answer>{expression or ''}</answer>")
    else:
        if not args.model:
            raise SystemExit("--model is required unless --oracle-solver is set")
        completions = generate_with_model(args.model, [row["prompt"] for row in rows], args.max_new_tokens)

    evaluated = []
    solved = 0
    for row, completion in zip(rows, completions):
        result = validate_expression(completion, row["numbers"])
        solved += int(result.ok)
        evaluated.append(
            {
                **row,
                "completion": completion,
                "valid": result.ok,
                "value": float(result.value) if result.value is not None else None,
                "reason": result.reason,
                "expression": result.expression,
            }
        )

    solved_rate = solved / len(rows) if rows else 0.0
    print(json.dumps({"count": len(rows), "solved": solved, "solved_rate": solved_rate}, ensure_ascii=False))
    if args.output:
        write_jsonl(args.output, evaluated)


if __name__ == "__main__":
    main()

