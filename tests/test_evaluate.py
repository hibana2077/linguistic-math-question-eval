import os
import json
import unittest

from evaluate_openrouter import parse_answer, evaluate

DATA_EN = "data/sample_gsm8k_en.json"

class EvaluateTests(unittest.TestCase):
    def test_parse_answer(self):
        self.assertEqual(parse_answer("The answer is 42."), "42")
        self.assertIsNone(parse_answer("No numbers here."))

    def test_mock_evaluation(self):
        acc = evaluate(DATA_EN, model="mock", mock=True)
        self.assertEqual(acc, 1.0)

if __name__ == "__main__":
    unittest.main()
