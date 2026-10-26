import asyncio
import json
from datetime import datetime
from unittest.mock import patch

import app
import normalize
import pipeline
from scripts.smoke_run import run_smoke


def jsonl(*rows):
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)


def synthetic_row(
    timestamp,
    message,
    chat_type="PRIVATE",
    sender="Synthetic Counterparty",
    sender_number="1001",
    receiver_number="9000",
    **extra,
):
    return {
        "timestamp": timestamp,
        "chat_type": chat_type,
        "direction": "INCOMING",
        "sender_name": sender,
        "sender_number": sender_number,
        "receiver_number": receiver_number,
        "message": message,
        **extra,
    }


def test_parser_handles_dates_none_media_and_reparse_stable_ids():
    content = jsonl(
        synthetic_row("13/07/2026, 10:00:00 AM", "None"),
        synthetic_row("07/13/2026, 10:01:00 AM", "[image message]"),
    )

    first = pipeline.parse_raw_chat_content(content, chat_id="upload-a")
    second = pipeline.parse_raw_chat_content(content, chat_id="upload-b")

    assert [item["dt"] for item in first] == [
        datetime(2026, 7, 13, 10, 0),
        datetime(2026, 7, 13, 10, 1),
    ]
    assert first[0]["message_text"] == ""
    assert first[1]["message_text"] == ""
    assert first[1]["is_media"] is True
    assert first[1]["media_kind"] == "image"
    assert [item["message_id"] for item in first] == [
        item["message_id"] for item in second
    ]


def test_parser_separates_private_chats_and_keeps_renamed_group_key():
    rows = [
        synthetic_row("13/07/2026, 10:00:00 AM", "first", sender_number="1001"),
        synthetic_row("13/07/2026, 10:00:00 AM", "second", sender_number="2002"),
        synthetic_row(
            "13/07/2026, 10:01:00 AM",
            "group first",
            chat_type="GROUP",
            group_name="Previous group label",
            sender_number="1200123:3",
        ),
        synthetic_row(
            "13/07/2026, 10:02:00 AM",
            "group second",
            chat_type="GROUP",
            group_name="Updated group label",
            sender_number="1200123:8",
        ),
    ]

    parsed = pipeline.parse_raw_chat_content(jsonl(*rows), chat_id="upload")
    private = [item for item in parsed if item["chat_type"] == "PRIVATE"]
    group = [item for item in parsed if item["chat_type"] == "GROUP"]

    assert len({item["chat_key"] for item in private}) == 2
    assert group[0]["chat_key"] == group[1]["chat_key"] == "GROUP:1200123"


def test_noise_false_positives_and_acknowledgements_are_preserved():
    for message_text in ("5 cr left", "added to my bid at 8.1"):
        assert pipeline.get_noise_label({
            "sender": "Synthetic sender",
            "message_text": message_text,
        }) == "none"

    ack = {
        "message_id": "orphan-ack",
        "chat_key": "PRIVATE:1001",
        "chat_name": "Synthetic private",
        "chat_type": "PRIVATE",
        "sender": "Synthetic sender",
        "message_text": "done",
        "noise_label": pipeline.get_noise_label({
            "sender": "Synthetic sender",
            "message_text": "done",
        }),
        "dt": datetime(2026, 7, 13, 10, 0),
    }

    assert ack["noise_label"] == "ack"
    assert pipeline.assign_contexts([ack]) == []

    trade = dict(ack)
    trade.update({
        "message_id": "synthetic-trade",
        "message_text": "Buy 5 Cr GS at 8.25%",
        "noise_label": "none",
        "dt": datetime(2026, 7, 13, 9, 59),
    })
    attached_ack = dict(ack)
    attached_ack["dt"] = datetime(2026, 7, 13, 10, 0)
    trade_contexts = pipeline.assign_contexts([trade, attached_ack])
    assert len(trade_contexts) == 1
    assert trade_contexts[0]["message_ids"] == [
        "synthetic-trade",
        "orphan-ack",
    ]


def test_hints_cover_quantity_yield_and_known_isins():
    for text in ("5 L", "5 lakhs", "20-30 lakh"):
        hints = pipeline.compute_hints(text)
        assert hints["amount"], text

    assert "10%" in pipeline.compute_hints("10% offer")["yield"]
    assert any(
        value.startswith("@5.20")
        for value in pipeline.compute_hints("@5.20%")["yield"]
    )
    assert "8.33-32" in pipeline.compute_hints("8.33-32")["yield"]
    assert pipeline.compute_hints(
        "US0378331005 INE002A01018"
    )["valid_isins"] == ["US0378331005", "INE002A01018"]


def test_grouping_links_late_replies_and_nearest_media_but_not_orphan_ack():
    rows = [
        synthetic_row(
            "13/07/2026, 8:00:00 AM",
            "[image message]",
            sender_number="1001",
        ),
        synthetic_row(
            "13/07/2026, 9:00:00 AM",
            "[media message]",
            sender_number="1001",
        ),
        synthetic_row(
            "13/07/2026, 10:00:00 AM",
            "Buy 5 Cr GS at 8.25%",
            sender_number="1001",
        ),
        synthetic_row(
            "13/07/2026, 10:00:01 AM",
            "done",
            sender_number="1001",
        ),
        synthetic_row(
            "13/07/2026, 12:00:00 PM",
            "follow up",
            sender="Synthetic Desk",
            sender_number="9000",
            receiver_number="1001",
            direction="OUTGOING",
            replying_to="Buy 5 Cr GS at 8.25%",
        ),
        synthetic_row(
            "13/07/2026, 12:01:00 PM",
            "see attachment",
            sender="Synthetic Desk",
            sender_number="9000",
            receiver_number="1001",
            direction="OUTGOING",
            replying_to="[image message]",
        ),
    ]
    with patch.object(pipeline.config, "OWN_SENDER_NAMES", ["Synthetic Desk"]):
        messages = pipeline.parse_raw_chat_content(jsonl(*rows), chat_id="grouping")
    contexts = pipeline.assign_contexts(messages)

    late_reply = next(
        message for message in messages if message["message_text"] == "follow up"
    )
    late_context = next(
        context for context in contexts
        if late_reply["message_id"] in context["message_ids"]
    )
    media_reply = next(
        message for message in messages if message["message_text"] == "see attachment"
    )
    media_context = next(
        context for context in contexts
        if media_reply["message_id"] in context["message_ids"]
    )
    nearest_media = messages[1]
    earlier_media = messages[0]

    assert late_context["reply_linked_count"] >= 1
    assert late_context["end"] == datetime(2026, 7, 13, 12, 0)
    assert media_context["reply_linked_count"] == 1
    assert nearest_media["message_id"] in media_context["message_ids"]
    assert earlier_media["message_id"] not in media_context["message_ids"]

    orphan = dict(messages[-1])
    orphan["message_id"] = "orphan-ack"
    orphan["message_text"] = "ok"
    orphan["replying_to"] = ""
    orphan["noise_label"] = "ack"
    assert pipeline.assign_contexts([orphan]) == []


def test_normalizers_cover_quantities_settlement_and_absent_settlement():
    assert normalize.normalize_quantity("1.58 Cr")["amount_inr"] == 15_800_000
    assert normalize.normalize_quantity("88 L")["amount_inr"] == 8_800_000
    assert normalize.normalize_quantity(
        "158 units", face_value=100_000
    )["amount_inr"] == 15_800_000
    assert normalize.normalize_quantity(
        "4864 units", face_value=1_000
    )["amount_inr"] == 4_864_000
    assert normalize.resolve_settlement(
        "Mon VD", datetime(2026, 7, 8)
    )["settlement_date"] == "2026-07-13"
    assert normalize.resolve_settlement(
        "T+1", datetime(2026, 7, 10)
    )["settlement_date"] == "2026-07-13"
    assert normalize.resolve_settlement(
        "Tom T+0", datetime(2026, 7, 8)
    )["settlement_date"] == "2026-07-09"
    assert normalize.resolve_settlement(
        "", datetime(2026, 7, 8)
    )["settlement_date"] is None
    absent_settlement = pipeline.validate_extraction(
        {"status": "INITIATED"},
        "Chat: Synthetic\nType: PRIVATE\nDate: 2026-07-08\n"
        "#1 [10:00:00] Sender: trade inquiry",
    )
    assert absent_settlement["settlement_date"] is None
    assert absent_settlement["settlement_label"] is None


def test_validation_rejects_unsupported_isin_downgrades_and_never_sets_price():
    invalid_isin = pipeline.validate_extraction(
        {
            "isin_text": "US0378331006",
            "evidence": {"isin_text": ["#1"]},
        },
        "Chat: Synthetic\nType: PRIVATE\nDate: 2026-07-13\n"
        "#1 [10:00:00] Sender: US0378331006",
    )
    assert invalid_isin["isin_text"] is None
    assert invalid_isin["partial_isin_text"] == "US0378331006"
    assert "partial_isin" in invalid_isin["warnings"]

    concluded = pipeline.validate_extraction(
        {
            "bond_identity_text": "Synthetic bond",
            "quantity_text": "5 Cr",
            "ytm": "8.25",
            "price": None,
            "owner_side": "SELL",
            "status": "CONCLUDED",
            "status_evidence": "bid at 8.25%",
            "evidence": {
                "bond_identity_text": ["#1"],
                "quantity_text": ["#1"],
                "ytm": ["#1"],
                "owner_side": ["#1"],
                "status": ["#2"],
                "status_evidence": ["#1"],
            },
        },
        "Chat: Synthetic\nType: PRIVATE\nDate: 2026-07-13\n"
        "#1 [10:00:00] Sender: Synthetic bond 5 Cr bid at 8.25%\n"
        "#2 [10:01:00] Other: closed",
    )
    assert concluded["status"] != "CONCLUDED"
    assert "conclusion_downgraded" in concluded["warnings"]
    assert concluded["price"] is None


def test_mocked_multibond_extraction_persists_each_deal_message_link(tmp_path):
    rows = [
        synthetic_row(
            "13/07/2026, 10:00:00 AM",
            "US0378331005 5 Cr at 8.25%; "
            "INE002A01018 8 Cr at 8.50%",
            sender_number="1001",
        )
    ]
    upload_id = "synthetic-multibond"
    messages, parse_report = pipeline.parse_raw_chat_content(
        jsonl(*rows),
        chat_id=upload_id,
        return_report=True,
    )
    model_result = {
        "deals": [
            {
                "bond_identity_text": "US0378331005",
                "isin_text": "US0378331005",
                "quantity_text": "5 Cr",
                "ytm": "8.25",
                "status": "INITIATED",
                "evidence": {
                    "bond_identity_text": ["#1"],
                    "isin_text": ["#1"],
                    "quantity_text": ["#1"],
                    "ytm": ["#1"],
                    "status": ["#1"],
                },
            },
            {
                "bond_identity_text": "INE002A01018",
                "isin_text": "INE002A01018",
                "quantity_text": "8 Cr",
                "ytm": "8.50",
                "status": "INITIATED",
                "evidence": {
                    "bond_identity_text": ["#1"],
                    "isin_text": ["#1"],
                    "quantity_text": ["#1"],
                    "ytm": ["#1"],
                    "status": ["#1"],
                },
            },
        ],
        "prompt_version": "synthetic-test",
    }

    app._store_upload_session(upload_id, {
        "upload_id": upload_id,
        "parsed_messages": messages,
        "parse_report": parse_report,
        "extractions": [],
    })
    database_path = tmp_path / "synthetic.sqlite"
    try:
        with (
            patch.object(app.config, "MAX_CONTEXTS", None),
            patch.object(
                app.pipeline,
                "call_llm_extraction",
                return_value=(
                    model_result,
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
            ),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
            patch.object(app.db, "DB_PATH", str(database_path)),
        ):
            report = asyncio.run(app.extract_trades(upload_id))
            assert report["deals_by_status"] == {"INITIATED": 2}
            assert asyncio.run(app.store_extracted_data(upload_id))["transactions_stored"] == 2

        stored = app.db.fetch_database_data(str(database_path))
        assert len(stored["transactions"]) == 2
        assert all(
            transaction["message_ids"] == [messages[0]["message_id"]]
            for transaction in stored["transactions"]
        )
    finally:
        app.UPLOAD_SESSIONS.pop(upload_id, None)


def test_smoke_runner_returns_aggregate_extract_report_without_chat_text(
    tmp_path,
):
    input_path = tmp_path / "synthetic.jsonl"
    input_path.write_text(
        jsonl(synthetic_row(
            "13/07/2026, 10:00:00 AM",
            "US0378331005 5 Cr at 8.25%",
        )),
        encoding="utf-8",
    )
    model_result = {
        "deals": [{
            "isin_text": "ZZ0000000000",
            "status": "INITIATED",
            "evidence": {
                "isin_text": ["#1"],
                "status": ["#1"],
            },
        }],
        "prompt_version": "synthetic-test",
    }

    with (
        patch.object(app.config, "MAX_CONTEXTS", None),
        patch.object(
            app.pipeline,
            "call_llm_extraction",
            return_value=(
                model_result,
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
        ),
        patch.object(app.pipeline, "get_model_context_length", return_value=100),
    ):
        report = run_smoke(input_path)

    assert report["messages_parsed"] == 1
    assert report["contexts_detected"] == 1
    assert report["processed"] == 1
    assert report["deals_by_status"] == {"INITIATED": 1}
    assert report["token_usage"]["total_tokens"] == 2
    assert report["warnings_by_type"]["partial_isin"] == 1
    assert "extractions" not in report
    assert "synthetic" not in json.dumps(report).casefold()
