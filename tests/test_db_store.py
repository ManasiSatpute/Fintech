import json
import tempfile
import unittest
from pathlib import Path

import db


STATUSES = (
    "INITIATED",
    "NEGOTIATING",
    "CONCLUDED",
    "CANCELLED",
    "UNCERTAIN",
    "DISCARDED",
)


class DatabaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "store.db")
        db.init_db(self.db_path)
        self.message = {
            "message_id": "synthetic-message-1",
            "sender": "Synthetic Trader",
            "message_text": "Synthetic offer",
            "timestamp": "2026-10-26T10:00:00",
            "chat_key": "PRIVATE:synthetic",
            "chat_name": "Synthetic Chat",
            "chat_type": "PRIVATE",
            "direction": "INCOMING",
            "is_own": False,
            "is_media": False,
            "media_kind": None,
            "replying_to": "",
            "noise_label": "none",
            "sender_number": "+1-555-0100",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def deals(self):
        deals = []
        for index, status in enumerate(STATUSES):
            deals.append({
                "status": status,
                "context_id": f"context-{index}",
                "chat_key": "PRIVATE:synthetic",
                "bond_identity_text": None,
                "isin_text": None,
                "buyer_text": None,
                "seller_text": None,
                "quantity_text": None,
                "normalized_quantity": None,
                "ytm": None,
                "ytm_is_two_way": None,
                "price": None,
                "settlement_text": None,
                "settlement_label": None,
                "settlement_date": None,
                "maturity_text": None,
                "maturity_ym": None,
                "partial_isin_text": None,
                "rfq_id": None,
                "package_note": None,
                "evidence": {"status": ["#1"]},
                "warnings": [],
                "missing_fields": [],
                "confidence": None,
                "needs_review": None,
                "prompt_version": None,
                "model_name": None,
                "message_number_map": {"#1": self.message["message_id"]},
            })
        return deals

    def persist(self, deals):
        return db.persist_upload(
            self.db_path,
            "synthetic-upload",
            [self.message],
            deals,
            context_message_maps={
                deal["context_id"]: deal["message_number_map"]
                for deal in deals
            },
            prompt_version="synthetic-prompt-v1",
            model_name="synthetic-model",
        )

    def test_all_statuses_store_and_broadcast_message_links_to_multiple_deals(self):
        deals = self.deals()
        self.persist(deals)

        conn = db.get_db_connection(self.db_path)
        try:
            statuses = {
                row["status"]
                for row in conn.execute("SELECT status FROM transactions")
            }
            links_for_message = conn.execute(
                "SELECT COUNT(*) FROM transaction_messages WHERE message_id = ?",
                (self.message["message_id"],),
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(statuses, set(STATUSES))
        self.assertEqual(links_for_message, len(STATUSES))

    def test_null_fields_store_and_fetch_returns_new_fields(self):
        deal = self.deals()[0]
        deal["evidence"] = None
        deal["warnings"] = None
        deal["missing_fields"] = None
        self.persist([deal])

        result = db.fetch_database_data(self.db_path)
        stored = result["transactions"][0]

        self.assertIsNone(stored["buyer_text"])
        self.assertIsNone(stored["quantity_text"])
        self.assertIsNone(stored["amount_inr"])
        self.assertIsNone(stored["evidence_json"])
        self.assertIsNone(stored["warnings_json"])
        self.assertIsNone(stored["missing_fields_json"])
        self.assertEqual(stored["prompt_version"], "synthetic-prompt-v1")
        self.assertEqual(stored["model_name"], "synthetic-model")
        self.assertEqual(stored["message_ids"], [self.message["message_id"]])

    def test_fetch_returns_evidence_warnings_and_missing_fields_as_json(self):
        deal = self.deals()[0]
        deal["warnings"] = ["synthetic_warning"]
        deal["missing_fields"] = ["synthetic_field"]
        self.persist([deal])

        stored = db.fetch_database_data(self.db_path)["transactions"][0]

        self.assertEqual(json.loads(stored["evidence_json"]), {"status": ["#1"]})
        self.assertEqual(
            json.loads(stored["warnings_json"]),
            ["synthetic_warning"],
        )
        self.assertEqual(
            json.loads(stored["missing_fields_json"]),
            ["synthetic_field"],
        )

    def test_data_response_exposes_trade_review_and_settlement_fields(self):
        deal = self.deals()[0]
        deal.update({
            "ytm": 10.08,
            "settlement_text": "T+1",
            "settlement_label": "T1",
            "settlement_date": "2026-10-27",
            "needs_review": True,
            "confidence": 0.85,
        })
        self.persist([deal])

        transaction = db.fetch_database_data(self.db_path)["transactions"][0]

        self.assertEqual(transaction["ytm"], 10.08)
        self.assertEqual(transaction["ytm_display"], "10.08%")
        self.assertEqual(transaction["settlement_text"], "T+1")
        self.assertEqual(transaction["settlement_date"], "2026-10-27")
        self.assertEqual(transaction["status"], "INITIATED")
        self.assertEqual(transaction["needs_review"], 1)
        self.assertEqual(transaction["confidence"], 0.85)

    def test_non_numeric_ytm_text_survives_storage_and_data_formatting(self):
        deal = self.deals()[0]
        deal["ytm"] = "8.33-32"
        self.persist([deal])

        transaction = db.fetch_database_data(self.db_path)["transactions"][0]

        self.assertEqual(transaction["ytm"], "8.33-32")
        self.assertEqual(transaction["ytm_display"], "8.33-32")

    def test_missing_settlement_stores_null_not_t1(self):
        deal = self.deals()[0]
        self.persist([deal])

        stored = db.fetch_database_data(self.db_path)["transactions"][0]

        self.assertIsNone(stored["settlement_reference"])
        self.assertIsNone(stored["settlement_date"])
        self.assertEqual(stored["settlement_display"], "—")

    def test_mon_vd_stores_other_with_known_date(self):
        deal = self.deals()[0]
        deal["settlement_text"] = "Mon VD"
        deal["settlement_label"] = "OTHER"
        deal["settlement_date"] = "2026-10-26"
        self.persist([deal])

        stored = db.fetch_database_data(self.db_path)["transactions"][0]

        self.assertEqual(stored["settlement_reference"], "OTHER")
        self.assertEqual(stored["settlement_date"], "2026-10-26")
        self.assertEqual(stored["settlement_text"], "Mon VD")
        self.assertEqual(stored["settlement_display"], "Mon VD")

    def test_t_plus_one_stores_t1_with_known_date(self):
        deal = self.deals()[0]
        deal["settlement_text"] = "T+1"
        deal["settlement_label"] = "T1"
        deal["settlement_date"] = "2026-10-27"
        self.persist([deal])

        stored = db.fetch_database_data(self.db_path)["transactions"][0]

        self.assertEqual(stored["settlement_reference"], "T1")
        self.assertEqual(stored["settlement_date"], "2026-10-27")
        self.assertEqual(stored["settlement_display"], "T+1")

    def test_re_storing_upload_does_not_duplicate_messages(self):
        deals = self.deals()[:2]
        self.persist(deals)
        second_result = self.persist(deals)

        self.assertEqual(second_result["messages_stored"], 0)
        self.assertEqual(db.fetch_database_data(self.db_path)["counts"]["messages"], 1)

    def test_missing_and_unknown_statuses_are_stored_as_uncertain(self):
        missing = self.deals()[0]
        missing["status"] = None
        unknown = self.deals()[1]
        unknown["status"] = "NOT_A_STATUS"
        self.persist([missing, unknown])

        stored_statuses = {
            row["status"]
            for row in db.fetch_database_data(self.db_path)["transactions"]
        }

        self.assertEqual(stored_statuses, {"UNCERTAIN"})

    def test_deal_without_citations_links_all_messages_in_context(self):
        deal = self.deals()[0]
        deal["evidence"] = {}
        deal["message_number_map"] = {
            "#1": self.message["message_id"],
            "#2": "synthetic-message-2",
        }
        second_message = dict(self.message)
        second_message["message_id"] = "synthetic-message-2"
        second_message["timestamp"] = "2026-10-26T10:01:00"
        db.persist_upload(
            self.db_path,
            "synthetic-upload",
            [self.message, second_message],
            [deal],
            context_message_maps={
                deal["context_id"]: deal["message_number_map"],
            },
        )

        stored = db.fetch_database_data(self.db_path)["transactions"][0]
        self.assertEqual(
            set(stored["message_ids"]),
            {self.message["message_id"], "synthetic-message-2"},
        )


if __name__ == "__main__":
    unittest.main()
