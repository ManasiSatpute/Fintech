import re
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import db
from bond_matching import (
    BondFieldEvidence,
    BondMatchResult,
    BondMatchStatus,
)
from bond_reference import BondReferenceRecord
from isin_utils import IsinStatus, extract_isin_from_message


class IdentificationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "identification.db")
        db.init_db(self.db_path)
        self._insert_transaction("synthetic-transaction")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _insert_transaction(self, transaction_id, bond_id=None):
        conn = db.get_db_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, bond_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (transaction_id, bond_id, "synthetic-created", "synthetic-updated"),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _reference(isin="US0378331005"):
        return BondReferenceRecord(
            source_values={
                "ISIN": isin,
                "NAME_OF_ISSUER": "Synthetic Issuer",
                "SECURITY_DESCRIPTION": "Synthetic Security",
                "MATURITY_DATE": "14-Jan-2030",
                "INTEREST_RATE": "8.50%",
                "CURRENCY": "USD",
            },
            normalized_isin=isin,
            normalized_issuer_key="synthetic issuer",
            maturity_date=date(2030, 1, 14),
        )

    def _result(self, status, reference=None, evidence=True, isin="US0378331005"):
        source_text = "Synthetic Issuer"
        source_evidence = (
            (
                BondFieldEvidence(
                    field_name="issuer",
                    message_id="synthetic-message",
                    source_message_text=source_text,
                    start=0,
                    end=len(source_text),
                ),
            )
            if evidence
            else ()
        )
        isin_source_evidence = ()
        isin_status = None
        matching_method = "synthetic-method"
        if status == BondMatchStatus.PRIMARY_ISIN_MATCH:
            isin_result = extract_isin_from_message(
                f"ISIN {isin}", "synthetic-message"
            )
            isin_source_evidence = isin_result.source_evidence
            isin_status = IsinStatus.VALID_ISIN
            matching_method = "primary_isin"
        elif status == BondMatchStatus.SECONDARY_ISSUER_MATCH:
            matching_method = "secondary_issuer"
        return BondMatchResult(
            status=status,
            identity_level="synthetic-level",
            matching_method=matching_method,
            attempted_methods=(matching_method,),
            explanation="Synthetic evidence-based decision.",
            matched_reference=reference,
            candidate_references=(reference,) if reference is not None else (),
            isin_status=isin_status,
            normalized_isin=(
                reference.normalized_isin if reference is not None else None
            ),
            raw_issuer_text="Synthetic Issuer",
            normalized_issuer_key="synthetic issuer",
            source_evidence=source_evidence,
            isin_source_evidence=isin_source_evidence,
        )

    def _stored_transaction(self):
        conn = db.get_db_connection(self.db_path)
        try:
            return conn.execute(
                """
                SELECT bond_id, identification_status
                FROM transactions
                WHERE transaction_id = ?
                """,
                ("synthetic-transaction",),
            ).fetchone()
        finally:
            conn.close()

    def test_primary_match_creates_reference_bond_and_audit(self):
        result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )

        outcome = db.persist_identification_result(
            self.db_path, "synthetic-transaction", result
        )
        stored = self._stored_transaction()
        conn = db.get_db_connection(self.db_path)
        try:
            bond = conn.execute(
                "SELECT isin, issuer_name, maturity_date, coupon_rate, currency "
                "FROM bonds WHERE bond_id = ?",
                (stored["bond_id"],),
            ).fetchone()
            audit = conn.execute(
                "SELECT identification_status, matching_method, reason, "
                "raw_issuer_text, evidence_json FROM identification_audit"
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(outcome["identification_status"], "primary_isin_match")
        self.assertTrue(outcome["bond_linked"])
        self.assertEqual(stored["identification_status"], "primary_isin_match")
        self.assertEqual(bond["isin"], "US0378331005")
        self.assertEqual(bond["maturity_date"], "2030-01-14")
        self.assertEqual(bond["coupon_rate"], 8.5)
        self.assertEqual(bond["currency"], "USD")
        self.assertEqual(audit["matching_method"], "primary_isin")
        self.assertEqual(audit["raw_issuer_text"], "Synthetic Issuer")
        self.assertIn("Synthetic Issuer", audit["evidence_json"])

    def test_secondary_match_is_persisted(self):
        result = self._result(
            BondMatchStatus.SECONDARY_ISSUER_MATCH,
            self._reference(),
        )

        db.persist_identification_result(
            self.db_path, "synthetic-transaction", result
        )

        self.assertIsNotNone(self._stored_transaction()["bond_id"])

    def test_primary_status_without_valid_identifier_evidence_is_rejected(self):
        result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )
        result = BondMatchResult(
            **{
                **vars(result),
                "isin_status": None,
                "isin_source_evidence": (),
            }
        )

        with self.assertRaisesRegex(ValueError, "source evidence"):
            db.persist_identification_result(
                self.db_path, "synthetic-transaction", result
            )
        self.assertIsNone(self._stored_transaction()["bond_id"])

    def test_unmatched_and_ambiguous_results_leave_bond_link_null(self):
        for status in (
            BondMatchStatus.NO_MATCH,
            BondMatchStatus.AMBIGUOUS_MATCH,
            BondMatchStatus.CONFLICTING_EVIDENCE,
            BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
            BondMatchStatus.VALID_ISIN_NOT_FOUND,
        ):
            with self.subTest(status=status):
                self._insert_transaction(f"transaction-{status.value}")
                db.persist_identification_result(
                    self.db_path,
                    f"transaction-{status.value}",
                    self._result(status),
                )
                conn = db.get_db_connection(self.db_path)
                try:
                    stored = conn.execute(
                        "SELECT bond_id, identification_status FROM transactions "
                        "WHERE transaction_id = ?",
                        (f"transaction-{status.value}",),
                    ).fetchone()
                finally:
                    conn.close()
                self.assertIsNone(stored["bond_id"])
                self.assertEqual(stored["identification_status"], status.value)

    def test_failed_audit_write_rolls_back_bond_and_transaction_update(self):
        result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )

        with mock.patch(
            "db._insert_identification_audit",
            side_effect=sqlite3.OperationalError("synthetic audit failure"),
        ):
            with self.assertRaisesRegex(
                sqlite3.OperationalError, "synthetic audit failure"
            ):
                db.persist_identification_result(
                    self.db_path, "synthetic-transaction", result
                )

        stored = self._stored_transaction()
        conn = db.get_db_connection(self.db_path)
        try:
            bond_count = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0]
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM identification_audit"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertIsNone(stored["bond_id"])
        self.assertIsNone(stored["identification_status"])
        self.assertEqual(bond_count, 0)
        self.assertEqual(audit_count, 0)

    def test_later_identification_failure_rolls_back_entire_upload(self):
        parsed_messages = [{
            "message_id": "synthetic-upload-message",
            "sender": "Synthetic Sender",
            "message_text": "Synthetic upload message",
            "timestamp": "2026-10-26T10:00:00",
        }]
        deals = [
            {
                "chat_key": "synthetic-chat",
                "context_id": f"synthetic-context-{index}",
                "message_ids": ["synthetic-upload-message"],
                "message_indices": [0],
                "status": "INITIATED",
            }
            for index in range(2)
        ]
        result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )
        original_insert = db._insert_identification_audit
        write_count = 0

        def fail_second_audit(conn, audit_values):
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise sqlite3.OperationalError("synthetic later audit failure")
            return original_insert(conn, audit_values)

        with mock.patch(
            "db._insert_identification_audit",
            side_effect=fail_second_audit,
        ):
            with self.assertRaisesRegex(
                sqlite3.OperationalError,
                "synthetic later audit failure",
            ):
                db.persist_upload(
                    self.db_path,
                    "synthetic-upload",
                    parsed_messages,
                    deals,
                    source_hash="synthetic-upload-source",
                    defer_bond_creation=True,
                    identification_results=[result, result],
                )

        conn = db.get_db_connection(self.db_path)
        try:
            message_count = conn.execute(
                "SELECT COUNT(*) FROM messages "
                "WHERE message_id = 'synthetic-upload-message'"
            ).fetchone()[0]
            transaction_count = conn.execute(
                "SELECT COUNT(*) FROM transactions "
                "WHERE source_hash = 'synthetic-upload-source'"
            ).fetchone()[0]
            link_count = conn.execute(
                "SELECT COUNT(*) FROM transaction_messages"
            ).fetchone()[0]
            bond_count = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0]
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM identification_audit"
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(write_count, 2)
        self.assertEqual(message_count, 0)
        self.assertEqual(transaction_count, 0)
        self.assertEqual(link_count, 0)
        self.assertEqual(bond_count, 0)
        self.assertEqual(audit_count, 0)

    def test_changed_extraction_retry_does_not_relink_existing_transaction(self):
        parsed_messages = [{
            "message_id": "synthetic-retry-message",
            "sender": "Synthetic Sender",
            "message_text": "Synthetic retry message",
            "timestamp": "2026-10-26T10:00:00",
        }]
        original_deal = {
            "chat_key": "synthetic-chat",
            "context_id": "synthetic-retry-context",
            "message_ids": ["synthetic-retry-message"],
            "message_indices": [0],
            "status": "INITIATED",
            "price": 99.0,
        }
        original_result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )
        first = db.persist_upload(
            self.db_path,
            "synthetic-retry-upload",
            parsed_messages,
            [original_deal],
            source_hash="synthetic-retry-source",
            defer_bond_creation=True,
            identification_results=[original_result],
        )
        transaction_id = first["transaction_ids"][0]
        conn = db.get_db_connection(self.db_path)
        try:
            original = conn.execute(
                "SELECT bond_id, final_price FROM transactions "
                "WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
        finally:
            conn.close()

        changed_deal = {**original_deal, "price": 105.0}
        changed_result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference("US5949181045"),
            isin="US5949181045",
        )
        retry = db.persist_upload(
            self.db_path,
            "synthetic-retry-upload",
            parsed_messages,
            [changed_deal],
            source_hash="synthetic-retry-source",
            defer_bond_creation=True,
            identification_results=[changed_result],
        )

        self.assertTrue(retry["already_stored"])
        self.assertEqual(retry["transaction_ids"], [transaction_id])
        self.assertEqual(
            retry["identification_by_status"],
            {"primary_isin_match": 1},
        )
        self.assertEqual(retry["identification_skipped"], 1)
        conn = db.get_db_connection(self.db_path)
        try:
            stored = conn.execute(
                "SELECT bond_id, final_price FROM transactions "
                "WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM identification_audit "
                "WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(stored["bond_id"], original["bond_id"])
        self.assertEqual(stored["final_price"], 99.0)
        self.assertEqual(audit_count, 1)

        with self.assertRaisesRegex(ValueError, "different result requires"):
            db.persist_upload(
                self.db_path,
                "synthetic-retry-upload",
                parsed_messages,
                [changed_deal],
                source_hash="synthetic-retry-source",
                update_existing=True,
                defer_bond_creation=True,
                identification_results=[changed_result],
            )

        conn = db.get_db_connection(self.db_path)
        try:
            after_conflict = conn.execute(
                "SELECT bond_id, final_price FROM transactions "
                "WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
            bond_count = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0]
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM identification_audit "
                "WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(after_conflict["bond_id"], original["bond_id"])
        self.assertEqual(after_conflict["final_price"], 99.0)
        self.assertEqual(bond_count, 1)
        self.assertEqual(audit_count, 1)

    def test_retry_is_idempotent_for_identical_decision(self):
        result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )

        first = db.persist_identification_result(
            self.db_path, "synthetic-transaction", result
        )
        second = db.persist_identification_result(
            self.db_path, "synthetic-transaction", result
        )
        conn = db.get_db_connection(self.db_path)
        try:
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM identification_audit"
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertTrue(first["audit_inserted"])
        self.assertFalse(second["audit_inserted"])
        self.assertEqual(audit_count, 1)

    def test_conflicting_existing_link_is_not_overwritten(self):
        conn = db.get_db_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO bonds (
                    bond_id, isin, bond_identity, identity_type
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    "existing-bond",
                    "US5949181045",
                    "Synthetic Existing Bond",
                    "ISIN",
                ),
            )
            conn.execute(
                "UPDATE transactions SET bond_id = ? WHERE transaction_id = ?",
                ("existing-bond", "synthetic-transaction"),
            )
            conn.commit()
        finally:
            conn.close()

        result = self._result(
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            self._reference(),
        )
        with self.assertRaisesRegex(ValueError, "different bond link"):
            db.persist_identification_result(
                self.db_path, "synthetic-transaction", result
            )

        stored = self._stored_transaction()
        self.assertEqual(stored["bond_id"], "existing-bond")
        self.assertIsNone(stored["identification_status"])

    def test_unresolved_upload_does_not_create_unknown_bond(self):
        outcome = db.persist_upload(
            self.db_path,
            "synthetic-upload",
            [],
            [
                {
                    "chat_key": "synthetic-chat",
                    "message_ids": ["synthetic-message"],
                    "context_id": "synthetic-context",
                    "status": "INITIATED",
                }
            ],
        )
        conn = db.get_db_connection(self.db_path)
        try:
            null_bond_transactions = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE bond_id IS NULL"
            ).fetchone()[0]
            bond_count = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(outcome["transactions_inserted"], 1)
        self.assertEqual(null_bond_transactions, 2)
        self.assertEqual(bond_count, 0)


class NullableBondMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "version-4.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_version_4_rebuild_preserves_data_children_and_audit(self):
        version_4_sql = db.SCHEMA_SQL.replace(
            "bond_id                TEXT,",
            "bond_id                TEXT NOT NULL,",
            1,
        )
        version_4_sql = re.sub(
            r"CREATE TABLE IF NOT EXISTS identification_audit \(.*?\n\);",
            """
            CREATE TABLE IF NOT EXISTS identification_audit (
                audit_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                identification_status TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
            );
            """,
            version_4_sql,
            count=1,
            flags=re.DOTALL,
        )
        version_4_sql = re.sub(
            r"CREATE UNIQUE INDEX IF NOT EXISTS idx_identification_audit_decision.*?;",
            "",
            version_4_sql,
            count=1,
            flags=re.DOTALL,
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(version_4_sql)
        conn.execute(db.SCHEMA_VERSION_SQL)
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_transactions_source_context_deal
            ON transactions (source_hash, context_id, deal_index)
            """
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (4, "synthetic-v4"),
        )
        conn.execute(
            """
            INSERT INTO messages (
                message_id, chat_id, sender, message_text, timestamp,
                is_own, is_media
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "synthetic-message",
                "synthetic-chat",
                "Synthetic Sender",
                "Synthetic message",
                "synthetic-time",
                0,
                0,
            ),
        )
        conn.execute(
            """
            INSERT INTO bonds (bond_id, isin, bond_identity, identity_type)
            VALUES (?, ?, ?, ?)
            """,
            ("synthetic-bond", None, "Synthetic bond", "UNKNOWN"),
        )
        conn.execute(
            """
            INSERT INTO transactions (
                transaction_id, bond_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "synthetic-transaction",
                "synthetic-bond",
                "synthetic-created",
                "synthetic-updated",
            ),
        )
        conn.execute(
            "INSERT INTO transaction_messages VALUES (?, ?)",
            ("synthetic-transaction", "synthetic-message"),
        )
        conn.execute(
            """
            INSERT INTO identification_audit (
                audit_id, transaction_id, identification_status, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "synthetic-audit",
                "synthetic-transaction",
                "synthetic-status",
                "synthetic-created",
            ),
        )
        conn.commit()
        conn.close()

        db.init_db(self.db_path)

        conn = db.get_db_connection(self.db_path)
        try:
            bond_column = next(
                row for row in conn.execute("PRAGMA table_info(transactions)")
                if row["name"] == "bond_id"
            )
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "messages",
                    "bonds",
                    "transactions",
                    "transaction_messages",
                    "identification_audit",
                )
            }
            transaction = conn.execute(
                "SELECT transaction_id, bond_id FROM transactions"
            ).fetchone()
            audit = conn.execute(
                "SELECT audit_id, identification_status FROM identification_audit"
            ).fetchone()
            version = conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
            foreign_key_violations = conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            transaction_indexes = {
                row["name"] for row in conn.execute("PRAGMA index_list(transactions)")
            }
        finally:
            conn.close()

        self.assertEqual(bond_column["notnull"], 0)
        self.assertEqual(counts, {
            "messages": 1,
            "bonds": 1,
            "transactions": 1,
            "transaction_messages": 1,
            "identification_audit": 1,
        })
        self.assertEqual(transaction["transaction_id"], "synthetic-transaction")
        self.assertEqual(transaction["bond_id"], "synthetic-bond")
        self.assertEqual(audit["audit_id"], "synthetic-audit")
        self.assertEqual(audit["identification_status"], "synthetic-status")
        self.assertEqual(version, 5)
        self.assertEqual(foreign_key_violations, [])
        self.assertIn(
            "idx_transactions_source_context_deal",
            transaction_indexes,
        )

    def test_nullable_bond_foreign_key_still_enforces_non_null_relationships(self):
        db.init_db(self.db_path)
        conn = db.get_db_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, bond_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                ("synthetic-null-transaction", None, "created", "updated"),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO transactions (
                        transaction_id, bond_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        "synthetic-orphan-transaction",
                        "missing-bond",
                        "created",
                        "updated",
                    ),
                )
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
