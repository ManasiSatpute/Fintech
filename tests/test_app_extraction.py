import asyncio
import threading
import unittest
from unittest.mock import Mock, patch

import app


def context(context_id, start, snippet):
    return {
        "context_id": context_id,
        "chat_key": "CHAT:test",
        "chat_name": "Synthetic",
        "chat_type": "GROUP",
        "message_ids": [f"message-{context_id}"],
        "start": start,
        "group_name": "Synthetic",
        "messages": [],
        "message_indices": [],
        "snippet_text": snippet,
        "message_number_map": {"#1": f"message-{context_id}"},
    }


class AppExtractionTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        app.UPLOAD_SESSIONS.clear()

    async def test_llm_call_does_not_block_the_async_event_loop(self):
        app._store_upload_session("threaded-upload", {
            "upload_id": "threaded-upload",
            "parsed_messages": [],
            "parse_report": {},
        })
        llm_finished = threading.Event()
        loop_ran_during_llm_call = []

        def blocking_llm(_snippet):
            llm_finished.wait(timeout=1)
            llm_finished.set()
            return (
                {"deals": [], "prompt_version": "test"},
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        async def keep_event_loop_responsive():
            await asyncio.sleep(0.05)
            loop_ran_during_llm_call.append(not llm_finished.is_set())
            llm_finished.set()

        with (
            patch.object(
                app.pipeline,
                "group_into_trade_contexts",
                return_value=([context("slow", "2026-01-01", "snippet")], []),
            ),
            patch.object(
                app.pipeline,
                "call_llm_extraction",
                side_effect=blocking_llm,
            ),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
        ):
            await asyncio.gather(
                app.extract_trades("threaded-upload"),
                keep_event_loop_responsive(),
            )

        self.assertEqual(loop_ran_during_llm_call, [True])

    async def test_context_failures_are_isolated_and_reported(self):
        upload_id = "synthetic-upload"
        app._store_upload_session(upload_id, {
            "upload_id": upload_id,
            "parsed_messages": [{"is_system": True}],
            "parse_report": {"media": 2, "skipped": 3},
        })
        contexts = [
            context("later", "2026-01-02", "later snippet"),
            context("earlier", "2026-01-01", "earlier snippet"),
        ]
        llm_results = [
            (
                {
                    "extraction_failed": True,
                    "raw_text": "synthetic raw response",
                    "error": "invalid response",
                    "prompt_version": "test",
                },
                {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            ),
            (
                {
                    "deals": [{"status": "INITIATED"}],
                    "prompt_version": "test",
                },
                {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            ),
        ]
        llm_mock = Mock()
        llm_mock.side_effect = llm_results

        with (
            patch.object(app.config, "MAX_CONTEXTS", None),
            patch.object(
                app.pipeline,
                "group_into_trade_contexts",
                return_value=(contexts, []),
            ),
            patch.object(app.pipeline, "call_llm_extraction", llm_mock),
            patch.object(
                app.pipeline,
                "validate_extraction",
                return_value={"status": "INITIATED", "needs_review": True},
            ),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
        ):
            report = await app.extract_trades(upload_id)

        self.assertEqual(
            [call.args[0] for call in llm_mock.call_args_list],
            ["earlier snippet", "later snippet"],
        )
        self.assertEqual(report["contexts_detected"], 2)
        self.assertEqual(report["processed"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["messages_parsed"], 1)
        self.assertEqual(report["media"], 2)
        self.assertEqual(report["system"], 1)
        self.assertEqual(report["skipped_on_parse"], 3)
        self.assertEqual(report["deals_by_status"], {"INITIATED": 1})
        self.assertEqual(report["needs_review_count"], 1)
        self.assertEqual(
            report["context_errors"][0]["raw_llm_output"],
            "synthetic raw response",
        )
        deal = report["extractions"][1]
        self.assertEqual(deal["chat_key"], "CHAT:test")
        self.assertEqual(deal["chat_name"], "Synthetic")
        self.assertEqual(deal["chat_type"], "GROUP")
        self.assertEqual(deal["message_ids"], ["message-later"])

        with patch.object(
            app.db,
            "persist_upload",
            return_value={
                "messages_stored": 1,
                "bonds_stored": 0,
                "transactions_stored": 1,
            },
        ) as persist_upload:
            await app.store_extracted_data(upload_id)

        stored_deal = next(
            deal
            for deal in persist_upload.call_args.kwargs["validated_deals"]
            if not deal.get("extraction_failed")
        )
        self.assertEqual(stored_deal["message_ids"], ["message-later"])
        self.assertEqual(
            persist_upload.call_args.kwargs["context_message_maps"]["later"],
            {"#1": "message-later"},
        )

    async def test_empty_extraction_returns_the_complete_report_shape(self):
        app._store_upload_session("empty-upload", {
            "upload_id": "empty-upload",
            "parsed_messages": [],
            "parse_report": {"media": 0, "skipped": 2},
        })

        with (
            patch.object(
                app.pipeline,
                "group_into_trade_contexts",
                return_value=([], []),
            ),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
        ):
            report = await app.extract_trades("empty-upload")

        self.assertEqual(report["messages_parsed"], 0)
        self.assertEqual(report["media"], 0)
        self.assertEqual(report["system"], 0)
        self.assertEqual(report["skipped_on_parse"], 2)
        self.assertEqual(report["contexts_detected"], 0)
        self.assertEqual(report["skipped_low_relevance"], 0)
        self.assertEqual(report["processed"], 0)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["deals_by_status"], {})
        self.assertEqual(report["needs_review_count"], 0)
        self.assertEqual(
            report["token_usage"],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def test_masked_model_values_are_restored_after_validation(self):
        upload_id = "masked-upload"
        app._store_upload_session(upload_id, {
            "upload_id": upload_id,
            "parsed_messages": [],
            "parse_report": {},
        })
        masked_context = context("masked", "2026-01-01", "masked snippet")
        masked_context["privacy_map"] = {"[PII_001]": "Alice"}
        validation_inputs = []

        def validate(deal, context_snippet, message_datetime):
            validation_inputs.append((deal, context_snippet))
            return dict(deal)

        with (
            patch.object(
                app.pipeline,
                "group_into_trade_contexts",
                return_value=([masked_context], []),
            ),
            patch.object(
                app.pipeline,
                "call_llm_extraction",
                return_value=(
                    {
                        "deals": [{
                            "status": "INITIATED",
                            "buyer_text": "[PII_001]",
                        }],
                        "prompt_version": "test",
                    },
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
            ),
            patch.object(
                app.pipeline,
                "validate_extraction",
                side_effect=validate,
            ),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
        ):
            report = await app.extract_trades(upload_id)

        self.assertEqual(validation_inputs[0][0]["buyer_text"], "[PII_001]")
        self.assertEqual(validation_inputs[0][1], "masked snippet")
        self.assertEqual(report["extractions"][0]["buyer_text"], "Alice")

    def test_upload_session_expires_after_configured_ttl(self):
        with patch.object(app.config, "UPLOAD_SESSION_TTL_SECONDS", 5):
            with patch.object(app.time, "monotonic", return_value=100):
                app._store_upload_session("expired", {"upload_id": "expired"})
            with patch.object(app.time, "monotonic", return_value=106):
                self.assertIsNone(app._get_upload_session("expired"))
        self.assertNotIn("expired", app.UPLOAD_SESSIONS)


if __name__ == "__main__":
    unittest.main()
