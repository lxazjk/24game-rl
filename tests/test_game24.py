import unittest

from twenty_four_rl.game24 import extract_answer, has_r1_format, solve_24, validate_expression
from twenty_four_rl.evaluate import evaluate_completions
from twenty_four_rl.rewards import correctness_reward_func, format_reward_func


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

    def test_solver(self):
        expression = solve_24([1, 3, 4, 6])
        self.assertIsNotNone(expression)
        self.assertTrue(validate_expression(expression or "", [1, 3, 4, 6]).ok)


class RewardTest(unittest.TestCase):
    def test_rewards(self):
        completions = ["<think>x</think><answer>(1+3)*(6+0)</answer>"]
        self.assertEqual(format_reward_func(completions), [0.2])
        self.assertEqual(correctness_reward_func(completions, numbers=[[1, 3, 6, 0]]), [0.0])


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
        self.assertTrue(evaluated[0]["valid"])
        self.assertEqual(evaluated[0]["expression"], "6/(1-(3/4))")
        self.assertEqual(len(evaluated[0]["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
