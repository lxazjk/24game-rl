from __future__ import annotations

import re
from typing import Any, Sequence

from .game24 import analyze_expression, has_r1_format, validate_expression


def _numbers_from_kwargs(index: int, kwargs: dict[str, Any]) -> Sequence[int]:
    numbers = kwargs.get("numbers")
    if numbers is None:
        raise KeyError("reward function requires `numbers` in the dataset batch")
    if numbers and isinstance(numbers[0], (list, tuple)):
        return numbers[index]
    return numbers


def _target_from_kwargs(index: int, kwargs: dict[str, Any]) -> int:
    targets = kwargs.get("target")
    if targets is None:
        return 24
    if isinstance(targets, (list, tuple)):
        return int(targets[index])
    return int(targets)


def format_reward_func(completions: list[str], **_: Any) -> list[float]:
    return [0.2 if has_r1_format(completion) else 0.0 for completion in completions]


def format_light_reward_func(completions: list[str], **_: Any) -> list[float]:
    return [0.1 if has_r1_format(completion) else 0.0 for completion in completions]


def allowed_chars_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        rewards.append(0.1 if analysis.allowed_chars else 0.0)
    return rewards


def correctness_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        result = validate_expression(completion, numbers, target=target)
        rewards.append(1.0 if result.ok else 0.0)
    return rewards


def correctness_strong_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        result = validate_expression(completion, numbers, target=target)
        rewards.append(2.0 if result.ok else 0.0)
    return rewards


def expression_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        rewards.append(0.2 if analysis.parse_ok and analysis.value is not None else 0.0)
    return rewards


def expression_light_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        rewards.append(0.1 if analysis.parse_ok and analysis.value is not None else 0.0)
    return rewards


def number_usage_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        rewards.append(0.2 if analysis.numbers_ok else 0.0)
    return rewards


def number_usage_light_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        rewards.append(0.15 if analysis.numbers_ok else 0.0)
    return rewards


def closeness_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        if analysis.value is None:
            rewards.append(0.0)
            continue
        distance = abs(float(analysis.value) - float(target))
        scale = max(abs(float(target)), 1.0)
        reward = 0.3 * max(0.0, 1.0 - min(distance / scale, 1.0))
        if not analysis.numbers_ok:
            reward *= 0.5
        rewards.append(reward)
    return rewards


def closeness_light_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        numbers = _numbers_from_kwargs(idx, kwargs)
        target = _target_from_kwargs(idx, kwargs)
        analysis = analyze_expression(completion, numbers, target=target)
        if analysis.value is None:
            rewards.append(0.0)
            continue
        distance = abs(float(analysis.value) - float(target))
        scale = max(abs(float(target)), 1.0)
        reward = 0.1 * max(0.0, 1.0 - min(distance / scale, 1.0))
        if not analysis.numbers_ok:
            reward *= 0.5
        rewards.append(reward)
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
        rewards.append(0.1 if 3 <= len(answer) <= 80 and "=" not in answer else 0.0)
    return rewards


def structure_light_reward_func(completions: list[str], **_: Any) -> list[float]:
    """Smaller answer-shape reward for correctness-heavy GRPO."""
    rewards: list[float] = []
    for completion in completions:
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL | re.IGNORECASE)
        if not answer_match:
            rewards.append(0.0)
            continue
        answer = answer_match.group(1).strip()
        rewards.append(0.05 if 3 <= len(answer) <= 80 and "=" not in answer else 0.0)
    return rewards


def single_answer_reward_func(completions: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        answer_count = len(re.findall(r"<answer>.*?</answer>", completion or "", re.DOTALL | re.IGNORECASE))
        rewards.append(0.05 if answer_count == 1 else 0.0)
    return rewards


reward_funcs = [
    format_reward_func,
    structure_reward_func,
    expression_reward_func,
    number_usage_reward_func,
    closeness_reward_func,
    correctness_reward_func,
]


correctness_heavy_reward_funcs = [
    format_light_reward_func,
    structure_light_reward_func,
    single_answer_reward_func,
    allowed_chars_reward_func,
    expression_light_reward_func,
    number_usage_light_reward_func,
    closeness_light_reward_func,
    correctness_strong_reward_func,
]


REWARD_FUNC_SETS = {
    "shaped": reward_funcs,
    "correctness_heavy": correctness_heavy_reward_funcs,
}
