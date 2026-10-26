import sqlite3
import tempfile
import unittest
from pathlib import Path

import db


class DatabaseSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def synthetic_messages(self):
        return [
            {
                "message_id": "synthetic-message-1",
                "chat_id": "synthetic-upload",
                "sender": "Synthetic Trader",
                "message_text": "synthetic message",
                "timestamp": "2026-10-26T10:00:00",
                "chat_key": "PRIVATE:1001",
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
        ]

    def create_version_3_database(self, path, duplicate_isin=False):
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message_text TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE bonds (
                bond_id TEXT PRIMARY KEY,
                isin TEXT,
                bond_identity TEXT NOT NULL,
                identity_type TEXT NOT NULL
            );
            CREATE TABLE transactions (
                transaction_id TEXT PRIMARY KEY,
                source_hash TEXT,
                bond_id TEXT NOT NULL,
                buyer TEXT,
                status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                context_id TEXT,
                deal_index INTEGER,
                FOREIGN KEY (bond_id) REFERENCES bonds (bond_id),
                UNIQUE (source_hash, context_id, deal_index)
            );
            CREATE TABLE transaction_messages (
                transaction_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                PRIMARY KEY (transaction_id, message_id),
                FOREIGN KEY (transaction_id)
                    REFERENCES transactions (transaction_id),
                FOREIGN KEY (message_id) REFERENCES messages (message_id)
            );
            """
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (3, "synthetic timestamp"),
        )
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (
                "synthetic-message-id",
                "synthetic-chat-id",
                "Synthetic Trader",
                "synthetic message",
                "2026-10-26T10:00:00",
            ),
        )
        conn.executemany(
            """
            INSERT INTO bonds (bond_id, isin, bond_identity, identity_type)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("synthetic-bond-id", "XS0000000000", "Synthetic bond", "ISIN"),
                *(
                    [("synthetic-bond-id-duplicate", "XS0000000000",
                      "Another synthetic bond row", "ISIN")]
                    if duplicate_isin
                    else []
                ),
            ],
        )
        conn.execute(
            """
            INSERT INTO transactions (
                transaction_id, source_hash, bond_id, buyer, status,
                created_at, updated_at, context_id, deal_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "synthetic-transaction-id",
                "synthetic-source-hash",
                "synthetic-bond-id",
                "Synthetic Buyer",
                "NEGOTIATING",
                "2026-10-26T10:00:00",
                "2026-10-26T10:01:00",
                "synthetic-context-id",
                0,
            ),
        )
        conn.execute(
            "INSERT INTO transaction_messages VALUES (?, ?)",
            ("synthetic-transaction-id", "synthetic-message-id"),
        )
        conn.commit()
        conn.close()

    def test_schema_version_is_recorded(self):
        conn = db.get_db_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT version, applied_at FROM schema_version"
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(row["version"], db.SCHEMA_VERSION)
        self.assertTrue(row["applied_at"])

    def test_connections_enable_foreign_keys_and_use_explicit_transactions(self):
        conn = db.get_db_connection(self.db_path)
        try:
            foreign_keys_enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            isolation_level = conn.isolation_level
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO transaction_messages (transaction_id, message_id)
                    VALUES (?, ?)
                    """,
                    ("missing-transaction", "missing-message"),
                )
        finally:
            conn.close()

        self.assertEqual(foreign_keys_enabled, 1)
        self.assertEqual(isolation_level, "DEFERRED")

    def test_version_3_migration_preserves_rows_and_links_and_is_repeatable(self):
        legacy_path = str(Path(self.temp_dir.name) / "version-3.db")
        self.create_version_3_database(legacy_path)

        db.init_db(legacy_path)
        db.init_db(legacy_path)

        conn = db.get_db_connection(legacy_path)
        try:
            transaction = conn.execute(
                """
                SELECT transaction_id, bond_id, buyer, status, identification_status
                FROM transactions
                """
            ).fetchone()
            linked_message = conn.execute(
                "SELECT message_id FROM transaction_messages WHERE transaction_id = ?",
                ("synthetic-transaction-id",),
            ).fetchone()
            version = conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
            audit_foreign_keys = conn.execute(
                "PRAGMA foreign_key_list(identification_audit)"
            ).fetchall()
            isin_index = conn.execute(
                "PRAGMA index_list(bonds)"
            ).fetchall()
            counts = {
                table: conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                for table in ("messages", "bonds", "transactions", "transaction_messages")
            }
        finally:
            conn.close()

        self.assertEqual(version, db.SCHEMA_VERSION)
        self.assertEqual(transaction["transaction_id"], "synthetic-transaction-id")
        self.assertEqual(transaction["bond_id"], "synthetic-bond-id")
        self.assertEqual(transaction["buyer"], "Synthetic Buyer")
        self.assertEqual(transaction["status"], "NEGOTIATING")
        self.assertIsNone(transaction["identification_status"])
        self.assertEqual(linked_message["message_id"], "synthetic-message-id")
        self.assertEqual(counts, {
            "messages": 1,
            "bonds": 1,
            "transactions": 1,
            "transaction_messages": 1,
        })
        self.assertEqual(len(audit_foreign_keys), 1)
        self.assertTrue(any(
            index["name"] == "idx_bonds_isin_unique_not_null"
            and index["unique"]
            for index in isin_index
        ))

    def test_duplicate_isins_abort_migration_without_changing_existing_rows(self):
        legacy_path = str(Path(self.temp_dir.name) / "duplicate-isins.db")
        self.create_version_3_database(legacy_path, duplicate_isin=True)

        with self.assertRaisesRegex(RuntimeError, "duplicate ISIN group"):
            db.init_db(legacy_path)

        conn = sqlite3.connect(legacy_path)
        try:
            version = conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
            transaction_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(transactions)")
            }
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            counts = {
                table: conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                for table in ("messages", "bonds", "transactions", "transaction_messages")
            }
        finally:
            conn.close()

        self.assertEqual(version, 3)
        self.assertNotIn("identification_status", transaction_columns)
        self.assertNotIn("identification_audit", tables)
        self.assertEqual(counts, {
            "messages": 1,
            "bonds": 2,
            "transactions": 1,
            "transaction_messages": 1,
        })

    def test_messages_table_has_new_columns(self):
        conn = db.get_db_connection(self.db_path)
        try:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(messages)")
            }
        finally:
            conn.close()

        self.assertTrue({
            "chat_key",
            "chat_name",
            "chat_type",
            "direction",
            "is_own",
            "is_media",
            "media_kind",
            "replying_to_text",
            "noise_label",
            "sender_hash",
        }.issubset(columns))

    def test_new_schema_separates_identification_status_and_adds_audit_table(self):
        conn = db.get_db_connection(self.db_path)
        try:
            transaction_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(transactions)")
            }
            audit_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(identification_audit)")
            }
            indexes = conn.execute("PRAGMA index_list(bonds)").fetchall()
        finally:
            conn.close()

        self.assertIn("identification_status", transaction_columns)
        self.assertIn("status", transaction_columns)
        self.assertTrue({
            "audit_id",
            "transaction_id",
            "identification_status",
            "created_at",
        }.issubset(audit_columns))
        self.assertTrue(any(
            index["name"] == "idx_bonds_isin_unique_not_null"
            and index["unique"]
            for index in indexes
        ))

    def test_partial_isin_index_allows_nulls_and_rejects_duplicates(self):
        conn = db.get_db_connection(self.db_path)
        try:
            conn.executemany(
                """
                INSERT INTO bonds (bond_id, isin, bond_identity, identity_type)
                VALUES (?, ?, ?, ?)
                """,
                [
                    ("synthetic-null-bond-1", None, "Synthetic unknown 1", "UNKNOWN"),
                    ("synthetic-null-bond-2", None, "Synthetic unknown 2", "UNKNOWN"),
                    ("synthetic-isin-bond-1", "XS0000000000", "Synthetic ISIN", "ISIN"),
                ],
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO bonds (bond_id, isin, bond_identity, identity_type)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "synthetic-isin-bond-2",
                        "XS0000000000",
                        "Duplicate synthetic ISIN",
                        "ISIN",
                    ),
                )
        finally:
            conn.close()

    def test_reinserting_messages_does_not_duplicate_them(self):
        messages = self.synthetic_messages()
        conn = db.get_db_connection(self.db_path)
        try:
            first_count = db.insert_messages(conn, messages, "synthetic-upload")
            second_count = db.insert_messages(conn, messages, "synthetic-upload")
            stored_count = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        self.assertEqual(stored_count, 1)

    def test_sender_hash_is_not_the_raw_sender_number(self):
        sender_number = self.synthetic_messages()[0]["sender_number"]
        conn = db.get_db_connection(self.db_path)
        try:
            db.insert_messages(conn, self.synthetic_messages(), "synthetic-upload")
            sender_hash = conn.execute(
                "SELECT sender_hash FROM messages"
            ).fetchone()["sender_hash"]
        finally:
            conn.close()

        self.assertNotEqual(sender_hash, sender_number)
        self.assertEqual(len(sender_hash), 64)

    def test_reset_recreates_empty_schema(self):
        conn = db.get_db_connection(self.db_path)
        try:
            db.insert_messages(conn, self.synthetic_messages(), "synthetic-upload")
        finally:
            conn.close()

        db.reset_db(self.db_path)

        conn = db.get_db_connection(self.db_path)
        try:
            message_count = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            schema_row = conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            conn.close()

        self.assertEqual(message_count, 0)
        self.assertEqual(schema_row["version"], db.SCHEMA_VERSION)
        self.assertTrue({"messages", "bonds", "transactions"}.issubset(tables))


if __name__ == "__main__":
    unittest.main()
