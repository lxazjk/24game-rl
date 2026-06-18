from __future__ import annotations

import re
from typing import Any, Sequence

from .game24 import has_r1_format, validate_expression


def _numbers_from_kwargs(index: int, kwargs: dict[str, Any]) -> Sequence[int]:
    numbers = kwargs.get("numbers")
    if numbers is None:
        raise KeyError("reward function requires `numbers` in the dataset batch")
    if numbers and isinstance(numbers[0], (list, tuple)):
        return numbers[index]
    return numbers


def format_reward_func(completions: list[str], **_: Any) -> list[float]:
    return [0.2 if has_r1_format(completion) else 0.0 for completion in completions]


def correctness_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        result = validate_expression(completion, numbers)
        rewards.append(1.0 if result.ok else 0.0)
    return rewards


def structure_reward_func(completions: list[str], **_: Any) -> list[float]:
    """Small shaping reward for keeping the final answer compact."""
    rewards: list[float] = []
    for completion in completions:
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL | re.IGNORECASE)
        if not answer_match:
            rewards.append(0.0)
            continue
        answer = answer_match.group(1).strip()
        rewards.append(0.1 if 3 <= len(answer) <= 80 else 0.0)
    return rewards


reward_funcs = [format_reward_func, structure_reward_func, correctness_reward_func]
