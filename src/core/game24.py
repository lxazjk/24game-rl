from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence


answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
think_pattern = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
allowed_expression_pattern = re.compile(r"^[0-9+\-*/()\s]+$")
number_pattern = re.compile(r"\d+")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    value: Fraction | None = None
    reason: str = ""
    expression: str = ""


@dataclass(frozen=True)
class ExpressionAnalysis:
    expression: str
    has_answer: bool
    has_r1_format: bool
    allowed_chars: bool
    parse_ok: bool
    value: Fraction | None
    used_numbers: tuple[int, ...]
    numbers_ok: bool
    target_ok: bool
    valid: bool
    reason: str


def has_answer_tag(text: str) -> bool:
    """Check whether a response contains an <answer>...</answer> block."""
    return bool(answer_pattern.search(text or ""))


def extract_answer(text: str) -> str:
    """Return the expression inside <answer> tags, or the stripped text itself."""
    matches = answer_pattern.findall(text or "")
    if matches:
        return matches[-1].strip()
    return (text or "").strip()


def has_r1_format(text: str) -> bool:
    """Check whether a response follows the requested R1-style two-tag format."""
    if not text:
        return False
    return bool(think_pattern.search(text) and answer_pattern.search(text))


def _eval_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if node.value <= 0:
            raise ValueError("numbers must be positive integers")
        return Fraction(node.value, 1)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def analyze_expression(
    expression_or_response: str,
    numbers: Sequence[int],
    target: int = 24,
    tolerance: float = 1e-6,
) -> ExpressionAnalysis:
    """Analyze format, number use, parseability, and target correctness separately."""
    text = expression_or_response or ""
    expression = extract_answer(text)
    has_answer = has_answer_tag(text)
    r1_format = has_r1_format(text)
    used_numbers = tuple(int(token) for token in number_pattern.findall(expression))
    numbers_ok = Counter(used_numbers) == Counter(int(n) for n in numbers)

    if not expression:
        return ExpressionAnalysis(
            expression=expression,
            has_answer=has_answer,
            has_r1_format=r1_format,
            allowed_chars=False,
            parse_ok=False,
            value=None,
            used_numbers=used_numbers,
            numbers_ok=numbers_ok,
            target_ok=False,
            valid=False,
            reason="empty expression",
        )

    allowed_chars = bool(allowed_expression_pattern.fullmatch(expression))
    if not allowed_chars:
        return ExpressionAnalysis(
            expression=expression,
            has_answer=has_answer,
            has_r1_format=r1_format,
            allowed_chars=False,
            parse_ok=False,
            value=None,
            used_numbers=used_numbers,
            numbers_ok=numbers_ok,
            target_ok=False,
            valid=False,
            reason="contains disallowed characters",
        )

    value: Fraction | None = None
    parse_ok = False
    parse_reason = ""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_ast(tree)
        parse_ok = True
    except Exception as exc:
        parse_reason = str(exc)

    target_ok = False
    if value is not None:
        target_ok = math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=tolerance)

    valid = parse_ok and numbers_ok and target_ok
    if valid:
        reason = ""
    elif not parse_ok:
        reason = parse_reason
    elif not numbers_ok:
        reason = "numbers are not used exactly once"
    else:
        reason = f"value is {float(value):.12g}, expected {target}"

    return ExpressionAnalysis(
        expression=expression,
        has_answer=has_answer,
        has_r1_format=r1_format,
        allowed_chars=allowed_chars,
        parse_ok=parse_ok,
        value=value,
        used_numbers=used_numbers,
        numbers_ok=numbers_ok,
        target_ok=target_ok,
        valid=valid,
        reason=reason,
    )


def validate_expression(
    expression_or_response: str,
    numbers: Sequence[int],
    target: int = 24,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate that an expression uses exactly the given numbers and equals target."""
    analysis = analyze_expression(
        expression_or_response,
        numbers,
        target=target,
        tolerance=tolerance,
    )
    return ValidationResult(
        analysis.valid,
        value=analysis.value,
        reason=analysis.reason,
        expression=analysis.expression,
    )


@dataclass(frozen=True)
class _Candidate:
    value: Fraction
    expression: str


def solve_24(numbers: Sequence[int], target: int = 24) -> str | None:
    """Find one exact 24-game expression with a small exhaustive search."""
    candidates = [_Candidate(Fraction(int(n), 1), str(int(n))) for n in numbers]
    solution = _search(candidates, Fraction(target, 1))
    return solution.expression if solution else None


def is_solvable(numbers: Sequence[int], target: int = 24) -> bool:
    return solve_24(numbers, target=target) is not None


def _search(candidates: list[_Candidate], target: Fraction) -> _Candidate | None:
    if len(candidates) == 1:
        return candidates[0] if candidates[0].value == target else None

    seen_states: set[tuple[Fraction, ...]] = set()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            rest = [candidate for k, candidate in enumerate(candidates) if k not in (i, j)]

            for combined in _combine(a, b):
                next_candidates = rest + [combined]
                state = tuple(sorted(candidate.value for candidate in next_candidates))
                if state in seen_states:
                    continue
                seen_states.add(state)
                result = _search(next_candidates, target)
                if result is not None:
                    return result
    return None


def _combine(a: _Candidate, b: _Candidate) -> Iterable[_Candidate]:
    yield _Candidate(a.value + b.value, f"({a.expression}+{b.expression})")
    yield _Candidate(a.value - b.value, f"({a.expression}-{b.expression})")
    yield _Candidate(b.value - a.value, f"({b.expression}-{a.expression})")
    yield _Candidate(a.value * b.value, f"({a.expression}*{b.expression})")
    if b.value != 0:
        yield _Candidate(a.value / b.value, f"({a.expression}/{b.expression})")
    if a.value != 0:
        yield _Candidate(b.value / a.value, f"({b.expression}/{a.expression})")
