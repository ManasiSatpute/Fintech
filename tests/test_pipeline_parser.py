import json
import unittest
from datetime import datetime
from unittest.mock import patch

from scripts.baseline_report import pipeline


def jsonl(*rows):
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)


class JsonlParserTests(unittest.TestCase):
    def test_compute_hints_validates_isin_and_extracts_trade_signals(self):
        hints = pipeline.compute_hints(
            "US0378331005, ISIN 12345, 5-10 Cr, entire quantum, "
            "@5.20%, 8.33-32, 25 bps, lvl 7.1, ytm 8.2%, "
            "bids, lena hai, T+1, tomorrow, Monday, NCD, Samman"
        )

        self.assertTrue(hints["valid_isin"])
        self.assertEqual(hints["valid_isins"], ["US0378331005"])
        self.assertEqual(hints["invalid_isin_candidates"], [])
        self.assertEqual(hints["partial_isin"], "12345")
        self.assertTrue(any("5-10 Cr" in match for match in hints["amount"]))
        self.assertTrue(any("entire quantum" in match.lower() for match in hints["amount"]))
        self.assertTrue(any("8.33-32" in match for match in hints["yield"]))
        self.assertTrue(any("25 bps" in match for match in hints["yield"]))
        self.assertIn("bids", [match.lower() for match in hints["action"]])
        self.assertIn("lena hai", [match.lower() for match in hints["action"]])
        self.assertIn("T+1", hints["settlement"])
        self.assertIn("Monday", hints["settlement"])
        self.assertIn("NCD", hints["bond_vocabulary"])
        self.assertIn("Sammaan", hints["issuer_alias"])
        self.assertEqual(hints["relevance_score"], 11)

        invalid = pipeline.compute_hints("ISIN US0378331006")
        self.assertFalse(invalid["valid_isin"])
        self.assertEqual(invalid["invalid_isin_candidates"], ["US0378331006"])

    def test_isin_alone_does_not_pass_and_low_relevance_is_reported(self):
        message = {
            "sender": "Sender",
            "message_text": "US0378331005",
            "dt": datetime(2026, 7, 13, 10, 0),
            "group_name": "Desk",
            "chat_key": "GROUP:123",
            "chat_name": "Desk",
            "is_media": False,
            "is_system": False,
        }

        self.assertFalse(pipeline.is_bond_trade_relevant([message]))
        contexts, skipped = pipeline.group_into_trade_contexts(
            [message],
            return_skipped=True,
        )

        self.assertEqual(contexts, [])
        self.assertEqual(len(skipped), 1)
        self.assertLess(skipped[0]["relevance_score"], pipeline.config.RELEVANCE_THRESHOLD)

    def test_assign_contexts_groups_by_chat_key_and_follows_reply_across_gap(self):
        rows = [
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "direction": "INCOMING",
                "chat_type": "GROUP",
                "sender_name": "Trader A",
                "sender_number": "1200123",
                "group_name": "Old display name",
                "message": "Buy 5 cr GS 2030 at 7.5",
            },
            {
                "timestamp": "13/07/2026, 10:01:00 AM",
                "direction": "INCOMING",
                "chat_type": "GROUP",
                "sender_name": "Trader B",
                "sender_number": "1200456",
                "group_name": "Same display name",
                "message": "Buy 5 cr GS 2030 at 7.5",
            },
            {
                "timestamp": "13/07/2026, 12:00:00 PM",
                "direction": "OUTGOING",
                "chat_type": "GROUP",
                "sender_name": "Desk",
                "sender_number": "1200123",
                "group_name": "Renamed group",
                "message": "done",
                "replying_to": "Buy 5 cr GS 2030",
            },
            {
                "timestamp": "13/07/2026, 1:00:00 PM",
                "direction": "OUTGOING",
                "chat_type": "GROUP",
                "sender_name": "Desk",
                "sender_number": "1200123",
                "group_name": "Renamed group",
                "message": "yes",
            },
        ]
        messages = pipeline.parse_raw_chat_content(jsonl(*rows), "chat")
        contexts = pipeline.group_into_trade_contexts(messages)

        self.assertEqual(len(contexts), 2)
        earlier_chat_context = next(
            context for context in contexts if context["chat_key"] == "GROUP:1200123"
        )
        later_chat_context = next(
            context for context in contexts if context["chat_key"] == "GROUP:1200456"
        )
        self.assertEqual(earlier_chat_context["chat_name"], "Old display name")
        self.assertEqual(earlier_chat_context["reply_linked_count"], 1)
        self.assertEqual(earlier_chat_context["message_ids"], [
            messages[0]["message_id"],
            messages[2]["message_id"],
            messages[3]["message_id"],
        ])
        self.assertEqual(earlier_chat_context["start"], "2026-07-13T10:00:00")
        self.assertEqual(earlier_chat_context["end"], "2026-07-13T13:00:00")
        self.assertEqual(later_chat_context["chat_type"], "GROUP")
        self.assertEqual(later_chat_context["unresolved_replies"], [])

    def test_reply_placeholder_attaches_nearest_media_and_unmatched_reply_is_recorded(self):
        rows = [
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Counterparty",
                "sender_number": "1001",
                "message": "[image message]",
            },
            {
                "timestamp": "13/07/2026, 10:01:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Counterparty",
                "sender_number": "1001",
                "message": "Buy 5 cr GS 2030 at 7.5",
            },
            {
                "timestamp": "13/07/2026, 12:00:00 PM",
                "chat_type": "PRIVATE",
                "sender_name": "Desk",
                "sender_number": "9000",
                "receiver_number": "1001",
                "direction": "OUTGOING",
                "message": "see image",
                "replying_to": "[image message]",
            },
            {
                "timestamp": "13/07/2026, 1:00:00 PM",
                "chat_type": "PRIVATE",
                "sender_name": "Desk",
                "sender_number": "9000",
                "receiver_number": "1001",
                "direction": "OUTGOING",
                "message": "done",
                "replying_to": "no earlier message matches this text",
            },
        ]
        with patch.object(pipeline.config, "OWN_SENDER_NAMES", ["Desk"]):
            messages = pipeline.parse_raw_chat_content(jsonl(*rows), "chat")
        contexts, skipped = pipeline.group_into_trade_contexts(
            messages,
            return_skipped=True,
        )
        all_contexts = contexts + skipped
        self.assertEqual(len(all_contexts), 2)
        context = next(item for item in all_contexts if item["reply_linked_count"])
        unresolved_context = next(
            item for item in all_contexts if item["unresolved_replies"]
        )
        self.assertEqual(context["reply_linked_count"], 1)
        self.assertEqual(len(unresolved_context["unresolved_replies"]), 1)
        self.assertEqual(
            unresolved_context["unresolved_replies"][0]["message_id"],
            messages[3]["message_id"],
        )
        self.assertIn(messages[0]["message_id"], context["message_ids"])
        self.assertIn(messages[2]["message_id"], context["message_ids"])

    def test_ack_after_gap_does_not_create_an_independent_window(self):
        messages = [
            {
                "message_id": "old",
                "chat_key": "PRIVATE:1",
                "chat_name": "Private",
                "chat_type": "PRIVATE",
                "sender": "A",
                "message_text": "Buy 5 cr GS at 7.5",
                "noise_label": "none",
                "dt": datetime(2026, 7, 13, 10, 0),
            },
            {
                "message_id": "ack",
                "chat_key": "PRIVATE:1",
                "chat_name": "Private",
                "chat_type": "PRIVATE",
                "sender": "B",
                "message_text": "done",
                "noise_label": "ack",
                "dt": datetime(2026, 7, 13, 11, 0),
            },
            {
                "message_id": "new",
                "chat_key": "PRIVATE:1",
                "chat_name": "Private",
                "chat_type": "PRIVATE",
                "sender": "A",
                "message_text": "Another unrelated message",
                "noise_label": "none",
                "dt": datetime(2026, 7, 13, 11, 1),
            },
        ]
        windows = pipeline.assign_contexts(messages)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["message_ids"], ["old", "ack"])
        self.assertEqual(windows[1]["message_ids"], ["new"])
        self.assertEqual(windows, sorted(windows, key=lambda window: window["start"]))

    def test_messages_without_chat_key_are_not_partitioned_by_group_name(self):
        messages = [
            {
                "message_id": "first",
                "chat_id": "upload",
                "group_name": "Old name",
                "sender": "A",
                "message_text": "Buy 5 cr GS at 7.5",
                "dt": datetime(2026, 7, 13, 10, 0),
            },
            {
                "message_id": "second",
                "chat_id": "upload",
                "group_name": "New name",
                "sender": "B",
                "message_text": "done",
                "dt": datetime(2026, 7, 13, 10, 1),
            },
        ]

        windows = pipeline.assign_contexts(messages)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["message_ids"], ["first", "second"])

    def test_system_detection_is_anchored_and_labels_are_generic(self):
        self.assertFalse(pipeline.is_system_or_media({
            "sender": "Sender",
            "message_text": "5 cr left with me",
        }))
        self.assertFalse(pipeline.is_system_or_media({
            "sender": "Sender",
            "message_text": "5 cr left",
        }))
        self.assertFalse(pipeline.is_system_or_media({
            "sender": "Sender",
            "message_text": "added to my bid at 10.1",
        }))
        self.assertTrue(pipeline.is_system_or_media({
            "sender": "Sender",
            "message_text": "Counterparty added Another person",
        }))
        self.assertTrue(pipeline.is_system_or_media({
            "sender": "Chat",
            "message_text": "ordinary text",
        }))
        self.assertTrue(pipeline.is_system_or_media({
            "sender": "Sender",
            "message_text": (
                "Messages and calls are end-to-end encrypted. "
                "Only people in this chat can read, listen to, or share them."
            ),
        }))
        self.assertTrue(pipeline.is_system_or_media({
            "sender": "Sender",
            "message_text": "Your security code with another user changed.",
        }))
        self.assertEqual(
            pipeline.get_noise_label({"sender": "Sender", "message_text": "done"}),
            "ack",
        )
        self.assertEqual(
            pipeline.get_noise_label({"sender": "Sender", "message_text": "Hello there"}),
            "greeting",
        )
        self.assertEqual(
            pipeline.get_noise_label({"sender": "Sender", "message_text": "Hi team"}),
            "greeting",
        )
        for acknowledgement in ("ok", "okay", "yes", "sure", "done", "closed", "s", "np", "good", "haan"):
            self.assertEqual(
                pipeline.get_noise_label({
                    "sender": "Sender",
                    "message_text": acknowledgement,
                }),
                "ack",
            )
        self.assertEqual(
            pipeline.get_noise_label({"sender": "Sender", "message_text": "How are you?"}),
            "chitchat",
        )

    def test_ack_and_media_stay_in_trade_context_with_attachment_marker(self):
        rows = [
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "direction": "INCOMING",
                "chat_type": "GROUP",
                "sender_name": "Counterparty",
                "sender_number": "1200123",
                "group_name": "Desk",
                "message": "Buy 5 cr GS 2030 at 7.5",
            },
            {
                "timestamp": "13/07/2026, 10:00:01 AM",
                "direction": "INCOMING",
                "chat_type": "GROUP",
                "sender_name": "Counterparty",
                "sender_number": "1200123",
                "group_name": "Desk",
                "message": "[MEDIA message]",
            },
            {
                "timestamp": "13/07/2026, 10:00:02 AM",
                "direction": "OUTGOING",
                "chat_type": "GROUP",
                "sender_name": "Desk user",
                "sender_number": "1200123",
                "group_name": "Desk",
                "message": "done",
            },
        ]
        messages = pipeline.parse_raw_chat_content(jsonl(*rows), "chat")

        self.assertEqual([item["noise_label"] for item in messages], ["none", "none", "ack"])
        self.assertTrue(messages[1]["is_media"])
        self.assertEqual(messages[1]["message_text"], "")
        contexts = pipeline.group_into_trade_contexts(messages)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(len(contexts[0]["messages"]), 3)
        self.assertIn("[attachment]", contexts[0]["snippet_text"])
        self.assertNotIn("[MEDIA message]", contexts[0]["snippet_text"])
        self.assertIn("done", contexts[0]["snippet_text"])

    def test_preserves_fields_and_builds_stable_chat_identity(self):
        rows = [
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "direction": "INCOMING",
                "chat_type": "PRIVATE",
                "sender_name": "Counterparty Name",
                "sender_number": "1001:2",
                "receiver_number": "9000:1",
                "message": "Reply body",
                "replying_to": "[quoted body]",
            },
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "direction": "OUTGOING",
                "chat_type": "PRIVATE",
                "sender_name": "Desk User",
                "sender_number": "9000:3",
                "receiver_number": "1001:9",
                "message": "Reply body",
                "replying_to": None,
            },
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "direction": "INCOMING",
                "chat_type": "GROUP",
                "sender_name": "Counterparty Name",
                "sender_number": "1200123:4",
                "receiver_number": "9000",
                "group_name": "  Desk   Group  ",
                "message": "group message",
            },
            {
                "timestamp": "13/07/2026, 10:01:00 AM",
                "direction": "INCOMING",
                "chat_type": "GROUP",
                "sender_name": "Other Person",
                "sender_number": "1200123:8",
                "receiver_number": "9000",
                "group_name": "Renamed Desk Group",
                "message": "second group message",
            },
        ]
        with patch.object(pipeline.config, "OWN_SENDER_NAMES", ["Desk User"]):
            messages, report = pipeline.parse_raw_chat_content(
                jsonl(*rows),
                chat_id="upload-1",
                return_report=True,
            )

        private = [item for item in messages if item["chat_type"] == "PRIVATE"]
        groups = [item for item in messages if item["chat_type"] == "GROUP"]
        self.assertEqual(private[0]["chat_key"], "PRIVATE:1001")
        self.assertEqual(private[1]["chat_key"], "PRIVATE:1001")
        self.assertEqual(private[0]["chat_name"], "Counterparty Name")
        self.assertTrue(private[1]["is_own"])
        self.assertFalse(private[0]["is_own"])
        self.assertEqual(private[0]["replying_to"], "[quoted body]")
        self.assertEqual(private[1]["replying_to"], "")
        self.assertEqual(private[0]["direction"], "INCOMING")
        self.assertEqual(private[0]["sender_number"], "1001:2")
        self.assertEqual(groups[0]["chat_key"], "GROUP:1200123")
        self.assertEqual(groups[1]["chat_key"], "GROUP:1200123")
        self.assertEqual(groups[0]["chat_name"], "Desk Group")
        self.assertEqual(groups[0]["group_name"], "  Desk   Group  ")
        self.assertEqual(report["parsed"], 4)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(report["per_chat_type"], {"GROUP": 2, "PRIVATE": 2})
        self.assertEqual(report["distinct_chat_keys"], 2)

        repeated, _ = pipeline.parse_raw_chat_content(
            jsonl(*rows),
            chat_id="different-upload-id",
            return_report=True,
        )
        self.assertEqual(
            [item["message_id"] for item in messages],
            [item["message_id"] for item in repeated],
        )

    def test_cleans_text_marks_media_and_reports_skipped_timestamps(self):
        rows = [
            {
                "timestamp": "13/07/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "Ａ\u200bB\nC",
            },
            {
                "timestamp": "13/07/2026, 10:00:01 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": None,
            },
            {
                "timestamp": "13/07/2026, 10:00:02 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "None",
            },
            *[
                {
                    "timestamp": f"13/07/2026, 10:00:0{second} AM",
                    "chat_type": "PRIVATE",
                    "sender_name": "Sender",
                    "sender_number": "1001",
                    "message": placeholder,
                }
                for second, placeholder in enumerate(
                    ("[MEDIA message]", "[image message]", "[chat message]"),
                    start=3,
                )
            ],
            {
                "timestamp": "not a timestamp",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "not parsed",
            },
        ]
        content = jsonl(*rows) + "\nnot-json"
        messages, report = pipeline.parse_raw_chat_content(
            content,
            chat_id="chat",
            return_report=True,
        )

        self.assertEqual(messages[0]["message_text"], "AB\nC")
        self.assertEqual(messages[1]["message_text"], "")
        self.assertEqual(messages[2]["message_text"], "")
        self.assertEqual(
            [(item["is_media"], item["media_kind"]) for item in messages[3:]],
            [(True, "media"), (True, "image"), (True, "chat")],
        )
        self.assertTrue(all(item["message_text"] == "" for item in messages[3:]))
        self.assertEqual(report["parsed"], 6)
        self.assertEqual(report["skipped"], 2)
        self.assertEqual(report["media"], 3)

    def test_scans_file_for_date_order_and_preserves_tie_order(self):
        day_first_rows = [
            {
                "timestamp": "07/08/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "ambiguous first",
            },
            {
                "timestamp": "13/08/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "date evidence",
            },
            {
                "timestamp": "07/08/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "ambiguous second",
            },
        ]
        messages = pipeline.parse_raw_chat_content(jsonl(*day_first_rows), "chat")
        self.assertEqual(messages[0]["dt"], datetime(2026, 8, 7, 10))
        self.assertEqual(messages[0]["message_text"], "ambiguous first")
        self.assertEqual(messages[1]["message_text"], "ambiguous second")

        month_first_rows = [
            {
                "timestamp": "08/07/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "month-first",
            },
            {
                "timestamp": "08/13/2026, 10:00:00 AM",
                "chat_type": "PRIVATE",
                "sender_name": "Sender",
                "sender_number": "1001",
                "message": "date evidence",
            },
        ]
        month_first = pipeline.parse_raw_chat_content(jsonl(*month_first_rows), "chat")
        self.assertEqual(month_first[0]["dt"], datetime(2026, 8, 7, 10))

        with patch.object(pipeline.config, "DAY_FIRST_DEFAULT", False):
            defaulted = pipeline.parse_raw_chat_content(
                jsonl(month_first_rows[0]),
                "chat",
            )
        self.assertEqual(defaulted[0]["dt"], datetime(2026, 8, 7, 10))
        with patch.object(pipeline.config, "DAY_FIRST_DEFAULT", True):
            defaulted_day_first = pipeline.parse_raw_chat_content(
                jsonl(month_first_rows[0]),
                "chat",
            )
        self.assertEqual(defaulted_day_first[0]["dt"], datetime(2026, 7, 8, 10))

    def test_plain_text_uses_single_upload_chat_key_and_skips_bad_dates(self):
        content = (
            "[13/07/2026, 10:00:00 AM] Sender: First\n"
            "[99/99/2026, 10:00:00 AM] Sender: Invalid date\n"
            "[13/07/2026, 10:00:01 AM] Sender: [image message]\n"
        )
        messages, report = pipeline.parse_raw_chat_content(
            content,
            chat_id="upload-7",
            return_report=True,
        )

        self.assertEqual({item["chat_key"] for item in messages}, {"CHAT:upload-7"})
        self.assertEqual(report["parsed"], 2)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(report["media"], 1)


if __name__ == "__main__":
    unittest.main()
