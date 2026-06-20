import unittest

from twenty_four_rl.aggregate_predictions import aggregate_predictions
from twenty_four_rl.data_diagnostics import summarize_data
from twenty_four_rl.game24 import (
    analyze_expression,
    extract_answer,
    has_answer_tag,
    has_r1_format,
    solve_24,
    validate_expression,
)
from twenty_four_rl.data import parse_numbers
from twenty_four_rl.evaluate import evaluate_completions
from twenty_four_rl.prompts import build_prompt
from twenty_four_rl.rewards import (
    closeness_reward_func,
    correctness_reward_func,
    expression_reward_func,
    format_reward_func,
    number_usage_reward_func,
    structure_reward_func,
    allowed_chars_reward_func,
    correctness_strong_reward_func,
    format_light_reward_func,
)


class Game24ValidationTest(unittest.TestCase):
    def test_valid_expression(self):
        result = validate_expression("<answer>(8/(3-(8/3)))</answer>", [8, 3, 8, 3])
        self.assertTrue(result.ok, result.reason)

    def test_rejects_reused_number(self):
        result = validate_expression("<answer>6*4</answer>", [6, 6, 4, 4])
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "numbers are not used exactly once")

    def test_rejects_disallowed_call(self):
        result = validate_expression("<answer>max(1,2,3,4)*6</answer>", [1, 2, 3, 4])
        self.assertFalse(result.ok)

    def test_extract_answer(self):
        self.assertEqual(extract_answer("<think>x</think><answer>(1+2+3)*4</answer>"), "(1+2+3)*4")

    def test_extract_answer_uses_last_answer(self):
        response = "<answer>1+1+1+1</answer><think>retry</think><answer>(1+2+3)*4</answer>"
        self.assertEqual(extract_answer(response), "(1+2+3)*4")

    def test_format(self):
        self.assertTrue(has_r1_format("<think>x</think><answer>y</answer>"))
        self.assertFalse(has_r1_format("<answer>y</answer>"))
        self.assertTrue(has_answer_tag("<answer>y</answer>"))

    def test_analysis_separates_parse_value_and_number_use(self):
        result = validate_expression("<answer>6*4</answer>", [6, 6, 4, 4])
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "numbers are not used exactly once")

        analysis = analyze_expression("<answer>6*4</answer>", [6, 6, 4, 4])
        self.assertTrue(analysis.parse_ok)
        self.assertEqual(float(analysis.value or 0), 24.0)
        self.assertTrue(analysis.target_ok)
        self.assertFalse(analysis.numbers_ok)
        self.assertFalse(analysis.valid)

    def test_solver(self):
        expression = solve_24([1, 3, 4, 6])
        self.assertIsNotNone(expression)
        self.assertTrue(validate_expression(expression or "", [1, 3, 4, 6]).ok)

    def test_arbitrary_target_and_three_numbers(self):
        response = "<answer>44+19+35</answer>"
        self.assertTrue(validate_expression(response, [44, 19, 35], target=98).ok)
        self.assertFalse(validate_expression(response, [44, 19, 35]).ok)
        expression = solve_24([44, 19, 35], target=98)
        self.assertIsNotNone(expression)
        self.assertTrue(validate_expression(expression or "", [44, 19, 35], target=98).ok)

    def test_target_prompt_and_flexible_number_parser(self):
        prompt = build_prompt([44, 19, 35], target=98)
        self.assertIn("Target: 98", prompt)
        self.assertIn("Numbers: 44, 19, 35", prompt)
        self.assertEqual(parse_numbers({"nums": [44, 19, 35]}, allowed_lengths=(3, 4)), [44, 19, 35])


class RewardTest(unittest.TestCase):
    def test_rewards(self):
        completions = ["<think>x</think><answer>(1+3)*(6+0)</answer>"]
        self.assertEqual(format_reward_func(completions), [0.2])
        self.assertEqual(correctness_reward_func(completions, numbers=[[1, 3, 6, 0]]), [0.0])

    def test_partial_rewards(self):
        completions = [
            "<think>x</think><answer>(12-5)*3*2</answer>",
            "<think>x</think><answer>(12-5)*3*2=42</answer>",
        ]
        numbers = [[2, 3, 5, 12], [2, 3, 5, 12]]

        self.assertEqual(expression_reward_func(completions, numbers=numbers), [0.2, 0.0])
        self.assertEqual(number_usage_reward_func(completions, numbers=numbers), [0.2, 0.0])
        self.assertGreater(closeness_reward_func(completions, numbers=numbers)[0], 0.0)
        self.assertEqual(structure_reward_func(completions), [0.1, 0.0])

    def test_partial_rewards_for_wrong_number_multiset(self):
        completions = ["<think>x</think><answer>6*4</answer>"]
        numbers = [[6, 6, 4, 4]]

        self.assertEqual(correctness_reward_func(completions, numbers=numbers), [0.0])
        self.assertEqual(expression_reward_func(completions, numbers=numbers), [0.2])
        self.assertEqual(number_usage_reward_func(completions, numbers=numbers), [0.0])
        self.assertGreater(closeness_reward_func(completions, numbers=numbers)[0], 0.0)

    def test_correctness_heavy_rewards(self):
        correct = ["<think>x</think><answer>12/(3-(5/2))</answer>"]
        wrong = ["<think>x</think><answer>(12-5)*3*2</answer>"]
        numbers = [[2, 3, 5, 12]]

        self.assertEqual(format_light_reward_func(correct), [0.1])
        self.assertEqual(allowed_chars_reward_func(correct, numbers=numbers), [0.1])
        self.assertEqual(correctness_strong_reward_func(correct, numbers=numbers), [2.0])
        self.assertEqual(correctness_strong_reward_func(wrong, numbers=numbers), [0.0])

    def test_rewards_use_target(self):
        completions = ["<think>x</think><answer>44+19+35</answer>"]
        numbers = [[44, 19, 35]]

        self.assertEqual(correctness_reward_func(completions, numbers=numbers), [0.0])
        self.assertEqual(correctness_reward_func(completions, numbers=numbers, target=[98]), [1.0])


class EvaluationTest(unittest.TestCase):
    def test_solved_at_n_uses_any_valid_attempt(self):
        rows = [{"numbers": [1, 3, 4, 6], "prompt": "unused"}]
        completions = [
            [
                "<think>x</think><answer>1+3+4+6</answer>",
                "<think>x</think><answer>6/(1-(3/4))</answer>",
            ]
        ]

        evaluated, summary = evaluate_completions(rows, completions)

        self.assertEqual(summary["solved"], 1)
        self.assertEqual(summary["pass_at_1_count"], 0)
        self.assertEqual(summary["solved_at_n_count"], 1)
        self.assertTrue(evaluated[0]["valid"])
        self.assertFalse(evaluated[0]["pass_at_1"])
        self.assertEqual(evaluated[0]["first_valid_sample_index"], 1)
        self.assertEqual(evaluated[0]["candidates_until_solution"], 2)
        self.assertEqual(evaluated[0]["expression"], "6/(1-(3/4))")
        self.assertEqual(len(evaluated[0]["attempts"]), 2)

    def test_empty_evaluation_summary(self):
        evaluated, summary = evaluate_completions([], [])

        self.assertEqual(evaluated, [])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["solved_rate"])
        self.assertTrue(summary["empty"])

    def test_evaluation_uses_target(self):
        rows = [{"numbers": [44, 19, 35], "target": 98, "prompt": "unused"}]
        completions = [["<think>x</think><answer>44+19+35</answer>"]]

        evaluated, summary = evaluate_completions(rows, completions)

        self.assertEqual(summary["solved"], 1)
        self.assertTrue(evaluated[0]["valid"])


class UtilityTest(unittest.TestCase):
    def test_aggregate_predictions(self):
        rows = [
            {"numbers": [1, 3, 4, 6], "prompt": "a"},
            {"numbers": [2, 3, 5, 12], "prompt": "b"},
        ]
        predictions = [
            [
                {"numbers": [1, 3, 4, 6], "completion": "<answer>1+3+4+6</answer>"},
                {"numbers": [2, 3, 5, 12], "completion": "<answer>12/(3-(5/2))</answer>"},
            ],
            [
                {"numbers": [1, 3, 4, 6], "completion": "<answer>6/(1-(3/4))</answer>"},
            ],
        ]

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, prediction_rows in enumerate(predictions):
                path = Path(directory) / f"predictions_{index}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    import json

                    for row in prediction_rows:
                        handle.write(json.dumps(row) + "\n")
                paths.append(path)

            aggregated, summary = aggregate_predictions(rows, paths)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["solved"], 2)
        self.assertTrue(aggregated[0]["valid"])
        self.assertTrue(aggregated[1]["valid"])

    def test_data_diagnostics(self):
        summary = summarize_data(
            [{"numbers": [1, 2, 3, 4]}],
            [{"numbers": [4, 3, 2, 1], "seen_in_train": True, "solvable": True}],
            [],
        )

        self.assertEqual(summary["negative_count"], 0)
        self.assertTrue(summary["negative_empty"])
        self.assertEqual(summary["eval_overlap_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
