from __future__ import annotations

from typing import Sequence


system_prompt = (
    "You solve the 24 game. Use each given number exactly once. "
    "Only use +, -, *, / and parentheses. Think briefly, then put the final "
    "expression inside <answer>...</answer>."
)


def build_prompt(numbers: Sequence[int]) -> str:
    nums = ", ".join(str(int(n)) for n in numbers)
    return (
        f"{system_prompt}\n\n"
        f"Numbers: {nums}\n"
        "Respond in this exact format:\n"
        "<think>reasoning steps</think><answer>expression</answer>"
    )


def build_chat_prompt(numbers: Sequence[int]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Numbers: "
                + ", ".join(str(int(n)) for n in numbers)
                + "\nReturn <think>...</think><answer>...</answer>."
            ),
        },
    ]
