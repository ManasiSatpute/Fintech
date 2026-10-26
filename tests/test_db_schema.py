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
