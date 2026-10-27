import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.baseline_report import pipeline


def response(content):
    return SimpleNamespace(
        content=content,
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        response_metadata={},
    )


class ExtractionSchemaTests(unittest.TestCase):
    def test_schema_fields_are_nullable_and_statuses_are_constrained(self):
        result = pipeline.ExtractionResult.model_validate({"deals": [{}]})
        deal = result.model_dump(mode="json")["deals"][0]
        self.assertEqual(deal["status"], None)
        self.assertEqual(deal["settlement_text"], None)
        self.assertIn("status", pipeline.DealExtraction.model_json_schema()["properties"])
        self.assertNotIn("INCOMPLETE", str(pipeline.DealExtraction.model_json_schema()))

    def test_structured_json_retries_with_validation_error_and_returns_prompt_version(self):
        valid_response = response(json.dumps({
            "deals": [
                {
                    "bond_identity_text": "7.20% Synthetic Bond 2030",
                    "status": "INITIATED",
                    "settlement_text": None,
                }
            ]
        }))
        llm = SimpleNamespace(
            invoke=Mock(
                side_effect=[response('{"deals": "not a list"}'), valid_response]
            )
        )
        with patch.object(pipeline, "ChatOllama", return_value=llm) as constructor:
            with patch.object(pipeline.config, "LLM_RETRIES", 1):
                result, tokens = pipeline.call_llm_extraction(
                    "Chat: Synthetic\nType: PRIVATE\n#1 [OWN] Desk: offer"
                )

        self.assertEqual(len(result["deals"]), 1)
        self.assertEqual(result["prompt_version"], pipeline.PROMPT_VERSION)
        self.assertEqual(result["deals"][0]["settlement_text"], None)
        self.assertEqual(tokens["attempts"], 2)
        self.assertEqual(tokens["total_tokens"], 30)
        constructor.assert_called_once()
        llm.invoke.assert_called()
        second_prompt = llm.invoke.call_args_list[1].args[0]
        self.assertIn("failed schema/JSON validation", second_prompt)
        self.assertIn("Compact conversation glossary", second_prompt)
        self.assertIn("Synthetic examples", second_prompt)

    def test_exhausted_parse_failure_does_not_expose_raw_model_text(self):
        raw_text = "not-json model response"
        llm = SimpleNamespace(invoke=Mock(return_value=response(raw_text)))
        with patch.object(pipeline, "ChatOllama", return_value=llm):
            with patch.object(pipeline.config, "LLM_RETRIES", 0):
                result, tokens = pipeline.call_llm_extraction("synthetic context")

        self.assertTrue(result["extraction_failed"])
        self.assertIsNone(result["raw_text"])
        self.assertNotIn(raw_text, str(result))
        self.assertEqual(result["error_category"], "invalid_model_response")
        self.assertTrue(result["correlation_id"])
        self.assertNotIn("DISCARDED", result)
        self.assertEqual(result["error"], "Model extraction failed. Retry or check the model service.")
        self.assertEqual(result["prompt_version"], pipeline.PROMPT_VERSION)
        self.assertEqual(tokens["attempts"], 1)

    def test_provider_exception_details_are_not_returned_or_retried_in_prompt(self):
        secret_like_error = (
            r"C:\synthetic\private\settings.env credential=SYNTHETIC_SECRET"
        )
        llm = SimpleNamespace(
            invoke=Mock(side_effect=RuntimeError(secret_like_error))
        )
        with patch.object(pipeline, "ChatOllama", return_value=llm):
            with patch.object(pipeline.config, "LLM_RETRIES", 1):
                result, tokens = pipeline.call_llm_extraction("synthetic context")

        self.assertTrue(result["extraction_failed"])
        self.assertIsNone(result["raw_text"])
        self.assertEqual(result["error_category"], "provider_error")
        self.assertTrue(result["correlation_id"])
        self.assertNotIn("SYNTHETIC_SECRET", str(result))
        self.assertNotIn(r"C:\synthetic", str(result))
        self.assertNotIn("SYNTHETIC_SECRET", llm.invoke.call_args_list[1].args[0])
        self.assertNotIn("DISCARDED", result)
        self.assertEqual(tokens["attempts"], 2)

    def test_legacy_validator_does_not_add_t1_or_december_31(self):
        deal = pipeline.validate_extraction({
            "bond_identity_text": "Synthetic Bond",
            "status": "INITIATED",
        })
        self.assertIsNone(deal["settlement_text"])
        self.assertIsNone(deal["maturity_text"])
        self.assertIsNone(deal["maturity_date"])
        self.assertNotIn("settlement_reference", deal)

    def test_prompt_contains_trading_semantics_and_four_examples(self):
        prompt = pipeline._extraction_prompt("synthetic conversation")
        self.assertIn("any bids?", prompt)
        self.assertIn("holding bid at X", prompt)
        self.assertIn("not acceptance", prompt)
        self.assertIn("specific offer or inquiry", prompt)
        self.assertIn("pass_or_refusal", prompt)
        self.assertEqual(len(pipeline.EXTRACTION_EXAMPLES), 10)
        self.assertIn(
            'A number after an ISIN joined by a hyphen (for example "<ISIN> - 43") is a unit count, never a yield.',
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
