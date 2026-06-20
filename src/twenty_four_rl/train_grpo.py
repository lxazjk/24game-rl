from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .rewards import REWARD_FUNC_SETS, reward_funcs


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 on 24 game with GRPO.")
    parser.add_argument("--config", default="configs/grpo_qwen_1_5b.yaml")
    parser.add_argument("--max-steps", type=int, help="Override training max_steps for smoke runs.")
    parser.add_argument("--limit-train-rows", type=int, help="Use only the first N training rows.")
    parser.add_argument("--limit-eval-rows", type=int, help="Use only the first N eval rows.")
    parser.add_argument("--output-dir", help="Override output_dir without editing the config file.")
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    train_rows = read_jsonl(config["data"]["train_path"])
    eval_path = config["data"].get("eval_path")
    eval_rows = read_jsonl(eval_path) if eval_path and Path(eval_path).exists() else None
    if args.limit_train_rows:
        train_rows = train_rows[: args.limit_train_rows]
    if args.limit_eval_rows and eval_rows:
        eval_rows = eval_rows[: args.limit_eval_rows]
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
        config["training"]["num_train_epochs"] = 1

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        **model_kwargs,
    )

    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        **config["training"],
    )
    reward_set_name = config.get("reward_set", "shaped")
    selected_reward_funcs = REWARD_FUNC_SETS.get(reward_set_name)
    if selected_reward_funcs is None:
        available = ", ".join(sorted(REWARD_FUNC_SETS))
        raise ValueError(f"unknown reward_set {reward_set_name!r}; available: {available}")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=selected_reward_funcs,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
    )
    trainer.train()
    trainer.save_model(Path(config["output_dir"]) / "checkpoint-final")


if __name__ == "__main__":
    main()
