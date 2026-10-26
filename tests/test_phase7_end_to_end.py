import csv
import io
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile

import app
import db
from bond_reference import BondReferenceError, BondReferenceIndex


REFERENCE_COLUMNS = (
    "ISIN",
    "NAME_OF_ISSUER",
    "SECURITY_DESCRIPTION",
    "MATURITY_DATE",
    "INTEREST_RATE",
    "CURRENCY",
)


class PhaseSevenEndToEndTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "phase-seven.db")
        db.init_db(self.db_path)
        self.reference_path = Path(self.temp_dir.name) / "reference.csv"
        with self.reference_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(REFERENCE_COLUMNS)
            writer.writerows((
                (
                    "US0378331005",
                    "Synthetic Issuer A",
                    "Synthetic Security A",
                    "14-Jan-2030",
                    "8.50%",
                    "USD",
                ),
                (
                    "US5949181045",
                    "Synthetic Issuer B",
                    "Synthetic Security B",
                    "15-Feb-2031",
                    "9.25%",
                    "USD",
                ),
                (
                    "US3030303030",
                    "Synthetic Issuer C",
                    "Synthetic Security C",
                    "16-Mar-2032",
                    "7.25%",
                    "USD",
                ),
            ))
        self.reference_index = BondReferenceIndex.load(self.reference_path)
        app.UPLOAD_SESSIONS.clear()

    def tearDown(self):
        app.UPLOAD_SESSIONS.clear()
        self.temp_dir.cleanup()

    @staticmethod
    def _deal(**fields):
        return {
            "status": "INITIATED",
            "evidence": {"status": ["#1"]},
            **fields,
        }

    async def _extract_all_cases(self):
        cases = [
            (
                "valid",
                "Trade US0378331005",
                self._deal(
                    isin_text="US0378331005",
                    evidence={"isin_text": ["#1"], "status": ["#1"]},
                ),
            ),
            (
                "invalid",
                "Trade US0378331006",
                self._deal(
                    isin_text="US0378331006",
                    evidence={"isin_text": ["#1"], "status": ["#1"]},
                ),
            ),
            (
                "issuer-conflict",
                "US0378331005 Synthetic Issuer B",
                self._deal(
                    isin_text="US0378331005",
                    issuer_text="Synthetic Issuer B",
                    evidence={
                        "isin_text": ["#1"],
                        "issuer_text": ["#1"],
                        "status": ["#1"],
                    },
                ),
            ),
            (
                "ambiguous",
                "US0378331005 or US5949181045",
                self._deal(
                    bond_identity_text="US0378331005 or US5949181045",
                    evidence={
                        "bond_identity_text": ["#1"],
                        "status": ["#1"],
                    },
                ),
            ),
            (
                "unresolved",
                "Synthetic discussion without a bond identifier",
                self._deal(),
            ),
            (
                "yield-price",
                "YTM 10.08%",
                self._deal(
                    price=10.08,
                    evidence={"price": ["#1"], "status": ["#1"]},
                ),
            ),
            (
                "coupon-price",
                "Coupon 8.50%",
                self._deal(
                    price=8.5,
                    evidence={"price": ["#1"], "status": ["#1"]},
                ),
            ),
            (
                "amount-price",
                "Trade amount 5 Cr",
                self._deal(
                    price=5,
                    evidence={"price": ["#1"], "status": ["#1"]},
                ),
            ),
            (
                "explicit-price",
                "Clean price: 99.50",
                self._deal(
                    price=99.5,
                    evidence={"price": ["#1"], "status": ["#1"]},
                ),
            ),
        ]
        upload_id = "synthetic-phase-seven-upload"
        parsed_messages = []
        contexts = []
        deals_by_snippet = {}
        case_message_ids = {}
        for index, (name, source, deal) in enumerate(cases):
            message_id = f"synthetic-message-{name}"
            case_message_ids[name] = message_id
            timestamp = f"2026-10-26T10:{index:02d}:00"
            message = {
                "message_id": message_id,
                "chat_id": upload_id,
                "sender": "Synthetic Sender",
                "message_text": source,
                "timestamp": timestamp,
                "chat_key": "PRIVATE:synthetic",
                "chat_name": "Synthetic Chat",
                "chat_type": "PRIVATE",
                "direction": "INCOMING",
                "is_own": False,
                "is_media": False,
                "dt": datetime.fromisoformat(timestamp),
            }
            snippet = f"#1 [10:{index:02d}:00] Synthetic Sender: {source}"
            parsed_messages.append(message)
            contexts.append({
                "context_id": f"synthetic-context-{name}",
                "chat_key": message["chat_key"],
                "chat_name": message["chat_name"],
                "chat_type": message["chat_type"],
                "message_ids": [message_id],
                "start": timestamp,
                "group_name": "Synthetic",
                "messages": [message],
                "message_indices": [index],
                "snippet_text": snippet,
                "message_number_map": {"#1": message_id},
            })
            deals_by_snippet[snippet] = deal

        app._store_upload_session(upload_id, {
            "upload_id": upload_id,
            "source_hash": "synthetic-phase-seven-source-hash",
            "parsed_messages": parsed_messages,
            "parse_report": {},
        })

        def extract(snippet):
            return (
                {
                    "deals": [deals_by_snippet[snippet]],
                    "prompt_version": "synthetic-phase-seven",
                },
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )

        with (
            patch.object(app.config, "MAX_CONTEXTS", None),
            patch.object(
                app.pipeline,
                "group_into_trade_contexts",
                return_value=(contexts, []),
            ),
            patch.object(app.pipeline, "call_llm_extraction", side_effect=extract),
            patch.object(app.pipeline, "get_model_context_length", return_value=100),
            patch.object(
                app,
                "get_bond_reference_index",
                return_value=self.reference_index,
            ),
            TestClient(app.app) as client,
        ):
            response = client.post(f"/extract/{upload_id}")
            self.assertEqual(response.status_code, 200)
            report = response.json()
        return upload_id, report, case_message_ids

    async def test_integrated_extraction_store_retry_and_rollback(self):
        upload_id, report, case_message_ids = await self._extract_all_cases()
        expected_statuses = {
            "valid": "primary_isin_match",
            "invalid": "invalid_or_unsupported_identifier",
            "issuer-conflict": "conflicting_evidence",
            "ambiguous": "ambiguous_match",
            "unresolved": "no_match",
            "yield-price": "no_match",
            "coupon-price": "no_match",
            "amount-price": "no_match",
            "explicit-price": "no_match",
        }
        deals_by_context = {
            deal["context_id"].removeprefix("synthetic-context-"): deal
            for deal in report["extractions"]
        }
        for name, expected in expected_statuses.items():
            extracted = deals_by_context[name]
            self.assertEqual(extracted["identification"]["status"], expected)
            self.assertTrue(extracted["identification"]["explanation"])

        for name in ("yield-price", "coupon-price", "amount-price"):
            self.assertIsNone(deals_by_context[name]["price"])
            self.assertIn("unsupported_price", deals_by_context[name]["warnings"])
        self.assertEqual(deals_by_context["explicit-price"]["price"], 99.5)

        with patch.object(app.db, "DB_PATH", self.db_path):
            with TestClient(app.app) as client:
                original_insert = db._insert_identification_audit
                audit_writes = 0

                def fail_later_audit(conn, values):
                    nonlocal audit_writes
                    audit_writes += 1
                    if audit_writes == 2:
                        raise sqlite3.OperationalError(
                            "synthetic identification write failure"
                        )
                    return original_insert(conn, values)

                with patch(
                    "db._insert_identification_audit",
                    side_effect=fail_later_audit,
                ):
                    failed_response = client.post(f"/store/{upload_id}")
                self.assertEqual(failed_response.status_code, 500)

                conn = db.get_db_connection(self.db_path)
                try:
                    for table in (
                        "messages",
                        "transactions",
                        "transaction_messages",
                        "bonds",
                        "identification_audit",
                    ):
                        self.assertEqual(
                            conn.execute(
                                f"SELECT COUNT(*) FROM {table}"
                            ).fetchone()[0],
                            0,
                            table,
                        )
                finally:
                    conn.close()

                stored_response = client.post(f"/store/{upload_id}")
                self.assertEqual(stored_response.status_code, 200)
                stored = stored_response.json()
                self.assertEqual(
                    stored["identification_by_status"],
                    {
                        "ambiguous_match": 1,
                        "conflicting_evidence": 1,
                        "invalid_or_unsupported_identifier": 1,
                        "no_match": 5,
                        "primary_isin_match": 1,
                    },
                )

                conn = db.get_db_connection(self.db_path)
                try:
                    rows = conn.execute(
                        """
                        SELECT t.identification_status, t.bond_id, t.final_price,
                               b.isin, m.message_id
                        FROM transactions t
                        JOIN transaction_messages tm
                          ON tm.transaction_id = t.transaction_id
                        JOIN messages m ON m.message_id = tm.message_id
                        LEFT JOIN bonds b ON b.bond_id = t.bond_id
                        """
                    ).fetchall()
                    audit_count = conn.execute(
                        "SELECT COUNT(*) FROM identification_audit"
                    ).fetchone()[0]
                    bond_count = conn.execute(
                        "SELECT COUNT(*) FROM bonds"
                    ).fetchone()[0]
                finally:
                    conn.close()

                rows_by_message = {
                    row["message_id"]: row for row in rows
                }
                self.assertEqual(len(rows_by_message), 9)
                self.assertEqual(audit_count, 9)
                self.assertEqual(bond_count, 1)
                self.assertIsNotNone(
                    rows_by_message[case_message_ids["valid"]]["bond_id"]
                )
                self.assertEqual(
                    rows_by_message[case_message_ids["valid"]]["isin"],
                    "US0378331005",
                )
                for name in (
                    "invalid",
                    "issuer-conflict",
                    "ambiguous",
                    "unresolved",
                    "yield-price",
                    "coupon-price",
                    "amount-price",
                    "explicit-price",
                ):
                    self.assertIsNone(
                        rows_by_message[case_message_ids[name]]["bond_id"]
                    )
                for name in ("yield-price", "coupon-price", "amount-price"):
                    self.assertIsNone(
                        rows_by_message[case_message_ids[name]]["final_price"]
                    )
                self.assertEqual(
                    rows_by_message[
                        case_message_ids["explicit-price"]
                    ]["final_price"],
                    99.5,
                )

                duplicate_response = client.post(f"/store/{upload_id}")
                self.assertEqual(duplicate_response.status_code, 200)
                duplicate = duplicate_response.json()
                self.assertTrue(duplicate["already_stored"])
                self.assertEqual(duplicate["transactions_skipped"], 9)
                self.assertEqual(duplicate["identification_skipped"], 9)
                self.assertIn("no records were changed", duplicate["message"])
                conn = db.get_db_connection(self.db_path)
                try:
                    self.assertEqual(
                        conn.execute(
                            "SELECT COUNT(*) FROM identification_audit"
                        ).fetchone()[0],
                        9,
                    )
                    self.assertEqual(
                        conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0],
                        1,
                    )
                    self.assertEqual(
                        conn.execute(
                            """
                            SELECT bond_id FROM transactions
                            WHERE transaction_id = (
                                SELECT transaction_id
                                FROM transaction_messages
                                WHERE message_id = ?
                            )
                            """,
                            (case_message_ids["valid"],),
                        ).fetchone()[0],
                        rows_by_message[case_message_ids["valid"]]["bond_id"],
                    )
                finally:
                    conn.close()

                index_response = client.get("/")
                self.assertEqual(index_response.status_code, 200)
                self.assertIn("Already stored", index_response.text)
                self.assertIn("This extraction was not applied", index_response.text)
                self.assertIn("Price (clean)", index_response.text)
                self.assertIn("Failure category", index_response.text)
                self.assertIn("Correlation ID", index_response.text)
                self.assertNotIn("Raw model output", index_response.text)

    async def test_api_errors_do_not_echo_sensitive_exception_details(self):
        upload_id = "synthetic-error-upload"
        app._store_upload_session(upload_id, {
            "upload_id": upload_id,
            "parsed_messages": [],
            "extractions": [],
            "identification_results": [],
        })
        sensitive_error = (
            r"C:\synthetic\private\settings.env credential=SYNTHETIC_SECRET"
        )
        with patch.object(app.db, "DB_PATH", self.db_path):
            with TestClient(app.app) as client:
                with patch.object(
                    app.db,
                    "persist_upload",
                    side_effect=RuntimeError(sensitive_error),
                ):
                    store_response = client.post(f"/store/{upload_id}")
                self.assertEqual(store_response.status_code, 500)
                self.assertEqual(
                    store_response.json()["detail"],
                    "Database storage failed.",
                )
                for secret in (
                    "private",
                    "settings.env",
                    "credential",
                    "SYNTHETIC_SECRET",
                ):
                    self.assertNotIn(secret.casefold(), store_response.text.casefold())

                with patch.object(
                    app.db,
                    "fetch_database_data",
                    side_effect=RuntimeError(sensitive_error),
                ):
                    data_response = client.get("/data")
                self.assertEqual(data_response.status_code, 500)
                self.assertEqual(
                    data_response.json()["detail"],
                    "Failed to fetch database data.",
                )
                self.assertNotIn("SYNTHETIC_SECRET", data_response.text)

                context = {
                    "context_id": "synthetic-error-context",
                    "chat_key": "PRIVATE:synthetic",
                    "chat_name": "Synthetic Chat",
                    "chat_type": "PRIVATE",
                    "message_ids": ["synthetic-error-message"],
                    "start": "2026-10-26T10:00:00",
                    "message_indices": [0],
                    "message_number_map": {"#1": "synthetic-error-message"},
                    "messages": [],
                    "snippet_text": "#1 Synthetic test message",
                    "privacy_map": {},
                }
                app._store_upload_session(upload_id, {
                    "upload_id": upload_id,
                    "parsed_messages": [],
                    "parse_report": {},
                })
                with (
                    patch.object(
                        app.pipeline,
                        "group_into_trade_contexts",
                        return_value=([context], []),
                    ),
                    patch.object(
                        app.pipeline,
                        "call_llm_extraction",
                        side_effect=RuntimeError(sensitive_error),
                    ),
                    patch.object(
                        app.pipeline,
                        "get_model_context_length",
                        return_value=100,
                    ),
                    patch.object(app, "get_bond_reference_index", return_value=None),
                    self.assertLogs(app.logger, level="WARNING") as captured,
                ):
                    extraction_response = client.post(f"/extract/{upload_id}")
                self.assertEqual(extraction_response.status_code, 200)
                extraction = extraction_response.json()
                self.assertEqual(
                    extraction["context_errors"][0]["error"],
                    "Extraction failed for this context.",
                )
                self.assertIsNone(
                    extraction["context_errors"][0]["raw_llm_output"]
                )
                self.assertTrue(
                    extraction["context_errors"][0]["correlation_id"]
                )
                self.assertNotIn(
                    "SYNTHETIC_SECRET",
                    "\n".join(captured.output),
                )
                self.assertIn(
                    extraction["context_errors"][0]["correlation_id"],
                    "\n".join(captured.output),
                )
                for secret in (
                    r"c:\synthetic\private",
                    "settings.env",
                    "credential",
                    "SYNTHETIC_SECRET",
                ):
                    self.assertNotIn(
                        secret.casefold(),
                        extraction_response.text.casefold(),
                    )

                reference_error = (
                    r"C:\synthetic\private\ISIN_DETAILS.csv "
                    "credential=SYNTHETIC_SECRET"
                )
                app._store_upload_session(upload_id, {
                    "upload_id": upload_id,
                    "parsed_messages": [],
                    "parse_report": {},
                })
                with (
                    patch.object(
                        app.pipeline,
                        "group_into_trade_contexts",
                        return_value=([context], []),
                    ),
                    patch.object(
                        app.pipeline,
                        "call_llm_extraction",
                        return_value=(
                            {"deals": [], "prompt_version": "synthetic"},
                            {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                        ),
                    ),
                    patch.object(
                        app.pipeline,
                        "get_model_context_length",
                        return_value=100,
                    ),
                    patch.object(
                        app,
                        "get_bond_reference_index",
                        side_effect=BondReferenceError(reference_error),
                    ),
                ):
                    reference_response = client.post(f"/extract/{upload_id}")
                self.assertEqual(reference_response.status_code, 200)
                reference_report = reference_response.json()
                self.assertEqual(
                    reference_report["bond_reference_error"],
                    "Bond reference data is unavailable. Check BOND_REFERENCE_CSV.",
                )
                for secret in (
                    r"c:\synthetic\private",
                    "ISIN_DETAILS.csv",
                    "credential",
                    "SYNTHETIC_SECRET",
                ):
                    self.assertNotIn(
                        secret.casefold(),
                        reference_response.text.casefold(),
                    )

                app._store_upload_session(upload_id, {
                    "upload_id": upload_id,
                    "parsed_messages": [],
                    "parse_report": {},
                })
                with (
                    patch.object(
                        app.pipeline,
                        "group_into_trade_contexts",
                        return_value=([context], []),
                    ),
                    patch.object(
                        app.pipeline,
                        "call_llm_extraction",
                        return_value=(
                            {
                                "deals": [{}],
                                "prompt_version": "synthetic",
                            },
                            {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                        ),
                    ),
                    patch.object(
                        app.pipeline,
                        "validate_extraction",
                        return_value={
                            "extraction_failed": True,
                            "raw_text": sensitive_error,
                            "error": sensitive_error,
                            "prompt_version": "synthetic",
                        },
                    ),
                    patch.object(
                        app.pipeline,
                        "get_model_context_length",
                        return_value=100,
                    ),
                    patch.object(app, "get_bond_reference_index", return_value=None),
                ):
                    invalid_deal_response = client.post(f"/extract/{upload_id}")
                self.assertEqual(invalid_deal_response.status_code, 200)
                invalid_deal = invalid_deal_response.json()["extractions"][0]
                self.assertEqual(
                    invalid_deal["error"],
                    "A model deal could not be validated.",
                )
                self.assertNotIn("raw_text", invalid_deal)
                self.assertNotIn(
                    "SYNTHETIC_SECRET",
                    invalid_deal_response.text,
                )

        upload = StarletteUploadFile(
            filename="synthetic-upload.txt",
            file=io.BytesIO(b"synthetic"),
        )
        with patch.object(upload, "read", new=AsyncMock(side_effect=RuntimeError(
            sensitive_error
        ))):
            with self.assertRaises(HTTPException) as raised:
                await app.upload_file(upload)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "Failed to read uploaded file.")
        self.assertNotIn("SYNTHETIC_SECRET", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
