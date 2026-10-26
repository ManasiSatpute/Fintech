import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import app
import db
from bond_reference import BondReferenceIndex


class ApplicationBondIdentificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "integrated.db")
        db.init_db(self.db_path)
        self.csv_path = Path(self.temp_dir.name) / "reference.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "ISIN",
                "NAME_OF_ISSUER",
                "SECURITY_DESCRIPTION",
                "MATURITY_DATE",
                "INTEREST_RATE",
                "CURRENCY",
            ))
            writer.writerow((
                "US0378331005",
                "Synthetic Issuer",
                "Synthetic Security",
                "14-Jan-2030",
                "8.50%",
                "USD",
            ))
        self.reference_index = BondReferenceIndex.load(self.csv_path)
        app.UPLOAD_SESSIONS.clear()

    def tearDown(self):
        app.UPLOAD_SESSIONS.clear()
        self.temp_dir.cleanup()

    async def test_extract_store_and_retry_persist_supported_match_and_audit(self):
        upload_id = "synthetic-identification-upload"
        source_text = "Synthetic trade US0378331005"
        message = {
            "message_id": "synthetic-message",
            "chat_id": upload_id,
            "sender": "Synthetic Sender",
            "message_text": source_text,
            "timestamp": "2026-10-26T10:00:00",
            "chat_key": "PRIVATE:synthetic",
            "chat_name": "Synthetic Chat",
            "chat_type": "PRIVATE",
            "direction": "INCOMING",
            "is_own": False,
            "is_media": False,
            "dt": datetime(2026, 10, 26, 10, 0),
        }
        context = {
            "context_id": "synthetic-context",
            "chat_key": message["chat_key"],
            "chat_name": message["chat_name"],
            "chat_type": message["chat_type"],
            "message_ids": [message["message_id"]],
            "start": "2026-10-26T10:00:00",
            "group_name": "Synthetic",
            "messages": [message],
            "message_indices": [0],
            "snippet_text": "#1 [10:00:00] Sender: " + source_text,
            "message_number_map": {"#1": message["message_id"]},
        }
        app._store_upload_session(upload_id, {
            "upload_id": upload_id,
            "source_hash": "synthetic-source-hash",
            "parsed_messages": [message],
            "parse_report": {},
        })

        extracted_deal = {
            "bond_identity_text": "US0378331005",
            "issuer_text": None,
            "coupon_rate": None,
            "price": None,
            "status": "INITIATED",
            "evidence": {"bond_identity_text": ["#1"]},
        }
        with (
            patch.object(app.db, "DB_PATH", self.db_path),
            patch.object(app.config, "MAX_CONTEXTS", None),
            patch.object(
                app.pipeline,
                "group_into_trade_contexts",
                return_value=([context], []),
            ),
            patch.object(
                app.pipeline,
                "call_llm_extraction",
                return_value=(
                    {"deals": [extracted_deal], "prompt_version": "synthetic"},
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                ),
            ),
            patch.object(
                app.pipeline,
                "validate_extraction",
                return_value=dict(extracted_deal),
            ),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
            patch.object(
                app,
                "get_bond_reference_index",
                return_value=self.reference_index,
            ),
        ):
            report = await app.extract_trades(upload_id)
            stored_once = await app.store_extracted_data(upload_id)
            stored_twice = await app.store_extracted_data(upload_id)

        self.assertEqual(
            report["extractions"][0]["identification"]["status"],
            "primary_isin_match",
        )
        self.assertIsNone(report["extractions"][0]["price"])
        self.assertEqual(
            stored_once["identification_by_status"],
            {"primary_isin_match": 1},
        )
        self.assertEqual(
            stored_twice["identification_by_status"],
            {"primary_isin_match": 1},
        )
        self.assertEqual(stored_twice["identification_skipped"], 1)
        conn = db.get_db_connection(self.db_path)
        try:
            transaction = conn.execute(
                "SELECT identification_status, bond_id, final_price "
                "FROM transactions"
            ).fetchone()
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM identification_audit"
            ).fetchone()[0]
            bond_count = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(transaction["identification_status"], "primary_isin_match")
        self.assertIsNotNone(transaction["bond_id"])
        self.assertIsNone(transaction["final_price"])
        self.assertEqual(audit_count, 1)
        self.assertEqual(bond_count, 1)


if __name__ == "__main__":
    unittest.main()
