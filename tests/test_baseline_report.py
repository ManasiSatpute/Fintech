import importlib.util
import os
import runpy
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "baseline_report.py"
SPEC = importlib.util.spec_from_file_location("baseline_report", SCRIPT_PATH)
baseline_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline_report)


def message(timestamp, group_name="room", sender="sender"):
    return {
        "dt": timestamp,
        "group_name": group_name,
        "sender": sender,
        "message_text": "message",
    }


class BaselineReportTests(unittest.TestCase):
    def test_default_timestamp_parsing_remains_day_first(self):
        timestamp = baseline_report.pipeline.parse_timestamp_str(
            "08/07/2026",
            "10:49:59 AM",
        )

        self.assertEqual(timestamp, datetime(2026, 7, 8, 10, 49, 59))

    def test_config_settings_can_be_overridden_by_environment(self):
        environment = {
            "OWN_SENDER_NAMES": " Desk A, Desk B ",
            "DAY_FIRST_DEFAULT": "false",
            "GAP_MINUTES": "20",
            "MAX_CONTEXTS": "none",
            "MODEL_NAME": "test-model",
            "LLM_RETRIES": "2",
            "MARKET_HOLIDAYS": "2026-01-01, 2026-12-25",
            "PII_MASKING_ENABLED": "off",
            "MASK_PERSON_NAMES": "yes",
            "RELEVANCE_THRESHOLD": "2",
            "REVIEW_CONFIDENCE_THRESHOLD": "0.8",
            "ALLOWED_ORIGINS": "http://localhost:9000",
        }
        with patch.dict(os.environ, environment):
            settings = runpy.run_path(str(SCRIPT_PATH.parents[1] / "config.py"))

        self.assertEqual(settings["OWN_SENDER_NAMES"], ["Desk A", "Desk B"])
        self.assertFalse(settings["DAY_FIRST_DEFAULT"])
        self.assertEqual(settings["GAP_MINUTES"], 20)
        self.assertIsNone(settings["MAX_CONTEXTS"])
        self.assertEqual(settings["MODEL_NAME"], "test-model")
        self.assertEqual(settings["LLM_RETRIES"], 2)
        self.assertEqual(settings["MARKET_HOLIDAYS"], ["2026-01-01", "2026-12-25"])
        self.assertFalse(settings["PII_MASKING_ENABLED"])
        self.assertTrue(settings["MASK_PERSON_NAMES"])
        self.assertEqual(settings["RELEVANCE_THRESHOLD"], 2)
        self.assertEqual(settings["REVIEW_CONFIDENCE_THRESHOLD"], 0.8)
        self.assertEqual(settings["ALLOWED_ORIGINS"], ["http://localhost:9000"])

    def test_window_counts_split_by_gap_and_exclude_system_messages(self):
        start = datetime(2026, 1, 1)
        messages = [
            message(start),
            message(start + timedelta(minutes=1)),
            message(start + timedelta(minutes=10), sender="system"),
            message(start + timedelta(minutes=20)),
            message(start + timedelta(minutes=21), "other-room"),
        ]

        with patch.object(
            baseline_report.pipeline,
            "is_system_message",
            side_effect=lambda item: item["sender"] == "system",
        ):
            with patch.object(
                baseline_report.pipeline,
                "is_bond_trade_relevant",
                side_effect=lambda window: len(window) > 1,
            ):
                dropped, formed, relevant = baseline_report._window_counts(messages)

        self.assertEqual((dropped, formed, relevant), (1, 2, 2))

    def test_context_selection_preserves_chronological_order_and_cap(self):
        contexts = [
            {
                "snippet_text": "",
                "messages": [{"n": index}],
            }
            for index in range(9)
        ]
        contexts.append({
            "snippet_text": "",
            "messages": [{"n": 9}, {"n": 10}],
        })

        selected = baseline_report._select_contexts(contexts)

        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, contexts[:8])


if __name__ == "__main__":
    unittest.main()
