from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


metric_names = [
    "train_loss",
    "eval_reward",
    "eval_rewards/correctness_reward_func/mean",
    "eval_rewards/format_reward_func/mean",
    "eval_rewards/structure_reward_func/mean",
    "eval_kl",
    "eval_entropy",
]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_metrics(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for entry in log_history:
        for name in metric_names:
            if name in entry:
                summary[name] = entry[name]
        if "step" in entry:
            summary["step"] = entry["step"]
        if "epoch" in entry:
            summary["epoch"] = entry["epoch"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a GRPO trainer_state.json file.")
    parser.add_argument("trainer_state", help="Path to checkpoint/trainer_state.json.")
    parser.add_argument("--output", help="Optional JSON summary output path.")
    args = parser.parse_args()

    state = load_json(args.trainer_state)
    summary = {
        "global_step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
        "train_batch_size": state.get("train_batch_size"),
        **latest_metrics(state.get("log_history", [])),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
