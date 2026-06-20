from __future__ import annotations

from typing import Sequence


system_prompt = (
    "You solve arithmetic target games. Use each given number exactly once. "
    "Only use +, -, *, / and parentheses. Think briefly, then put the final "
    "expression inside <answer>...</answer>. The answer must be only an expression, "
    "with no equals sign, result value, example, or explanation."
)


def build_prompt(numbers: Sequence[int], target: int = 24) -> str:
    nums = ", ".join(str(int(n)) for n in numbers)
    task_line = "Make 24." if int(target) == 24 else f"Target: {int(target)}"
    return (
        f"{system_prompt}\n\n"
        f"{task_line}\n"
        f"Numbers: {nums}\n"
        "Respond in this exact format:\n"
        "<think>reasoning steps</think><answer>expression only</answer>"
    )


def build_chat_prompt(numbers: Sequence[int], target: int = 24) -> list[dict[str, str]]:
    target_line = "Make 24." if int(target) == 24 else f"Target: {int(target)}"
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                target_line
                + "\nNumbers: "
                + ", ".join(str(int(n)) for n in numbers)
                + "\nReturn <think>...</think><answer>expression only</answer>."
            ),
        },
    ]
