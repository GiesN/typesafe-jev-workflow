import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from typesafe_sdk import ChoiceAnswer

from src.typesafe_ai_langgraph.typesafe_ai_langgraph_workflow import build_workflow

EXAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "data/mock_emails.json").read_text()
)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_both_intents_and_preserves_answer(self):
        for intent, destination in (
            ("invoice", "accounts_payable"), ("general", "general_inbox")
        ):
            with self.subTest(intent=intent):
                probabilities = {"invoice": 0.8, "general": 0.2}
                if intent == "general":
                    probabilities = {"invoice": 0.2, "general": 0.8}
                answer = ChoiceAnswer(
                    choice=intent, confidence=0.6, probabilities=probabilities
                )
                client = SimpleNamespace(system_one=AsyncMock(return_value=SimpleNamespace(
                    choices={"intent": answer}, model="jev-1.12"
                )))
                email = {**EXAMPLES[0]["email"], "expected_intent": "do-not-send"}
                result = await build_workflow(client).ainvoke({"email": email})
                self.assertEqual(result["destination"], destination)
                self.assertEqual(result["intent"], intent)
                self.assertEqual(result["confidence"], 0.6)
                self.assertEqual(result["probabilities"], probabilities)
                client.system_one.assert_awaited_once()
                request = client.system_one.call_args.kwargs
                self.assertEqual(set(request["state"]["email"]), {"sender", "subject", "body"})
                self.assertEqual(set(request["questions"]["intent"].criteria), {"invoice", "general"})

    async def test_api_error_is_not_routed_as_general(self):
        client = SimpleNamespace(system_one=AsyncMock(side_effect=RuntimeError("API unavailable")))
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            await build_workflow(client).ainvoke({"email": EXAMPLES[0]["email"]})

    async def test_unexpected_label_is_rejected(self):
        client = SimpleNamespace(system_one=AsyncMock(return_value=SimpleNamespace(
            choices={"intent": ChoiceAnswer(choice="other", confidence=1.0, probabilities={"other": 1.0})}
        )))
        with self.assertRaisesRegex(ValueError, "Unexpected intent"):
            await build_workflow(client).ainvoke({"email": EXAMPLES[0]["email"]})

    def test_mock_data_has_ten_unique_balanced_examples(self):
        self.assertEqual(len(EXAMPLES), 10)
        self.assertEqual(len({item["email"]["id"] for item in EXAMPLES}), 10)
        self.assertEqual(sum(item["expected_intent"] == "invoice" for item in EXAMPLES), 5)
        self.assertEqual(sum(item["expected_intent"] == "general" for item in EXAMPLES), 5)
        for item in EXAMPLES:
            self.assertEqual(set(item["email"]), {"id", "sender", "subject", "body"})
            self.assertTrue(all(isinstance(value, str) and value for value in item["email"].values()))


if __name__ == "__main__":
    unittest.main()
