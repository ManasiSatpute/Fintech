"""
Database module for Bond Trade Extraction App.
Implements the exact SQLite schema required:
- messages
- bonds
- transactions
"""

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bond_matching import BondMatchResult, BondMatchStatus
from bond_reference import normalize_issuer_key
from isin_utils import IsinStatus, normalize_isin_candidate
import config

DB_PATH = "bond_deals.db"
SCHEMA_VERSION = 5

# Exact DDL specified in requirements
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,     -- UUID
    chat_id         TEXT NOT NULL,
    sender          TEXT NOT NULL,
    message_text    TEXT NOT NULL,
    timestamp       TEXT NOT NULL,         -- ISO 8601 datetime
    chat_key        TEXT,
    chat_name       TEXT,
    chat_type       TEXT,
    direction       TEXT,
    is_own          INTEGER NOT NULL CHECK (is_own IN (0, 1)),
    is_media        INTEGER NOT NULL CHECK (is_media IN (0, 1)),
    media_kind      TEXT,
    replying_to_text TEXT,
    noise_label     TEXT,
    sender_hash     TEXT
);

CREATE TABLE IF NOT EXISTS bonds (
    bond_id         TEXT PRIMARY KEY,      -- UUID
    isin            TEXT,                  -- VARCHAR(12), nullable
    bond_identity   TEXT NOT NULL,
    identity_type   TEXT NOT NULL CHECK (identity_type IN
                        ('ISIN','CUSIP','SERIES','NAME_ONLY','UNKNOWN')),
    issuer_name     TEXT,
    credit_rating   TEXT,
    maturity_date   TEXT,                  -- ISO date
    coupon_rate     REAL,
    face_value      REAL,
    currency        TEXT                   -- VARCHAR(3), e.g. 'INR'
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id        TEXT PRIMARY KEY,   -- UUID
    source_hash            TEXT,
    bond_id                TEXT,
    buyer                   TEXT,
    seller                  TEXT,
    quantity                REAL,
    quantity_unit            TEXT,
    is_indicative             INTEGER,          -- 0/1 boolean
    ytm                        REAL,
    final_price                 REAL,
    settlement_reference          TEXT CHECK (
        settlement_reference IS NULL
        OR settlement_reference IN ('T0','T1','T2','OTHER')
    ),
    settlement_date                 TEXT,       -- ISO date
    status                            TEXT CHECK (
        status IS NULL OR status IN (
            'INITIATED','NEGOTIATING','CONCLUDED','CANCELLED','UNCERTAIN','DISCARDED'
        )
    ),
    identification_status             TEXT,
    created_at                          TEXT NOT NULL,
    updated_at                           TEXT NOT NULL,
    context_id                          TEXT,
    deal_index                          INTEGER,
    chat_key                            TEXT,
    owner_side                          TEXT,
    buyer_text                          TEXT,
    seller_text                         TEXT,
    quantity_text                       TEXT,
    amount_inr                          NUMERIC,
    units_assumed                       INTEGER,
    ytm_is_two_way                      INTEGER,
    settlement_text                     TEXT,
    maturity_text                       TEXT,
    maturity_ym                         TEXT,
    partial_isin_text                   TEXT,
    rfq_id                              TEXT,
    package_note                        TEXT,
    evidence_json                       TEXT,
    warnings_json                       TEXT,
    missing_fields_json                 TEXT,
    confidence                          REAL,
    needs_review                        INTEGER,
    prompt_version                      TEXT,
    model_name                          TEXT,
    FOREIGN KEY (bond_id) REFERENCES bonds (bond_id),
    UNIQUE (source_hash, context_id, deal_index)
);

CREATE TABLE IF NOT EXISTS transaction_messages (
    transaction_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    PRIMARY KEY (transaction_id, message_id),
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id),
    FOREIGN KEY (message_id) REFERENCES messages (message_id)
);

CREATE TABLE IF NOT EXISTS identification_audit (
    audit_id                 TEXT PRIMARY KEY,
    transaction_id           TEXT NOT NULL,
    identification_status    TEXT,
    matching_method          TEXT,
    reason                   TEXT,
    normalized_isin          TEXT,
    raw_issuer_text          TEXT,
    normalized_issuer_key    TEXT,
    evidence_json            TEXT,
    candidates_json          TEXT,
    decision_key             TEXT,
    created_at               TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_identification_audit_decision
    ON identification_audit (transaction_id, decision_key);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bonds_isin_unique_not_null
    ON bonds (isin)
    WHERE isin IS NOT NULL;
"""

SCHEMA_VERSION_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER NOT NULL,
    applied_at   TEXT NOT NULL
);
"""


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path,
        timeout=30.0,
        isolation_level="DEFERRED",
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        foreign_keys_enabled = conn.execute("PRAGMA foreign_keys").fetchone()
        if foreign_keys_enabled is None or foreign_keys_enabled[0] != 1:
            raise RuntimeError("SQLite foreign-key enforcement could not be enabled.")
    except Exception:
        conn.close()
        raise
    return conn


def _migration_3(conn: sqlite3.Connection) -> None:
    transaction_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(transactions)")
    }
    if "source_hash" not in transaction_columns:
        conn.execute("ALTER TABLE transactions ADD COLUMN source_hash TEXT")
    if "deal_index" not in transaction_columns:
        conn.execute("ALTER TABLE transactions ADD COLUMN deal_index INTEGER")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_source_context_deal
        ON transactions (source_hash, context_id, deal_index)
        """
    )


def _migration_4(conn: sqlite3.Connection) -> None:
    existing_tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "transactions" not in existing_tables or "bonds" not in existing_tables:
        raise RuntimeError(
            "Cannot apply SQLite migration 4: required transactions or bonds "
            "table is missing."
        )

    transaction_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(transactions)")
    }
    if "identification_status" not in transaction_columns:
        conn.execute(
            "ALTER TABLE transactions ADD COLUMN identification_status TEXT"
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identification_audit (
            audit_id TEXT PRIMARY KEY,
            transaction_id TEXT NOT NULL,
            identification_status TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (transaction_id)
                REFERENCES transactions (transaction_id)
        )
        """
    )

    duplicate_isin_groups = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT isin
            FROM bonds
            WHERE isin IS NOT NULL
            GROUP BY isin
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicate_isin_groups:
        raise RuntimeError(
            "Cannot create the unique non-null ISIN index: "
            f"{duplicate_isin_groups} duplicate ISIN group(s) exist. "
            "No bond or transaction rows were changed; resolve duplicates "
            "manually using the documented procedure, then retry."
        )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_bonds_isin_unique_not_null
        ON bonds (isin)
        WHERE isin IS NOT NULL
        """
    )


def _migration_5(conn: sqlite3.Connection) -> None:
    transaction_sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("transactions",),
    ).fetchone()
    if transaction_sql_row is None or not transaction_sql_row["sql"]:
        raise RuntimeError(
            "Cannot apply SQLite migration 5: transactions table definition is missing."
        )

    transaction_sql = transaction_sql_row["sql"]
    nullable_sql, replacements = re.subn(
        r"(?im)^(\s*b ond_id\s+TEXT)\s+NOT\s+NULL\b".replace(" ", ""),
        r"\1",
        transaction_sql,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(
            "Cannot apply SQLite migration 5: transactions.bond_id is not "
            "a recognized NOT NULL column."
        )

    dependent_schema = conn.execute(
        """
        SELECT type, sql
        FROM sqlite_master
        WHERE tbl_name = ? AND type IN ('index', 'trigger') AND sql IS NOT NULL
        """,
        ("transactions",),
    ).fetchall()
    old_table_name = re.compile(r"(?i)\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?transactions\b")
    replacement_sql, table_replacements = old_table_name.subn(
        "CREATE TABLE transactions_phase5_new",
        nullable_sql,
        count=1,
    )
    if table_replacements != 1:
        raise RuntimeError(
            "Cannot apply SQLite migration 5: transactions table could not be "
            "safely rebuilt."
        )

    conn.execute(replacement_sql)
    column_names = [
        row["name"] for row in conn.execute("PRAGMA table_info(transactions)")
    ]
    columns_sql = ", ".join(f'"{name}"' for name in column_names)
    conn.execute(
        f"INSERT INTO transactions_phase5_new ({columns_sql}) "
        f"SELECT {columns_sql} FROM transactions"
    )
    conn.execute("DROP TABLE transactions")
    conn.execute(
        "ALTER TABLE transactions_phase5_new RENAME TO transactions"
    )
    for schema_object in dependent_schema:
        conn.execute(schema_object["sql"])

    audit_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(identification_audit)")
    }
    audit_additions = {
        "matching_method": "TEXT",
        "reason": "TEXT",
        "normalized_isin": "TEXT",
        "raw_issuer_text": "TEXT",
        "normalized_issuer_key": "TEXT",
        "evidence_json": "TEXT",
        "candidates_json": "TEXT",
        "decision_key": "TEXT",
    }
    for column, column_type in audit_additions.items():
        if column not in audit_columns:
            conn.execute(
                f"ALTER TABLE identification_audit "
                f"ADD COLUMN {column} {column_type}"
            )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_identification_audit_decision
        ON identification_audit (transaction_id, decision_key)
        """
    )

    bond_id_column = next(
        (
            row for row in conn.execute("PRAGMA table_info(transactions)")
            if row["name"] == "bond_id"
        ),
        None,
    )
    if bond_id_column is None or bond_id_column["notnull"]:
        raise RuntimeError(
            "SQLite migration 5 did not make transactions.bond_id nullable."
        )
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(
            "SQLite migration 5 detected foreign-key violations; migration rolled back."
        )


MIGRATIONS = {
    3: _migration_3,
    4: _migration_4,
    5: _migration_5,
}


def init_db(db_path: str = DB_PATH) -> None:
    """Create a new schema or apply versioned, transactional SQLite migrations."""
    conn = get_db_connection(db_path)
    try:
        existing_tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not existing_tables:
            try:
                conn.executescript("BEGIN IMMEDIATE;\n" + SCHEMA_SQL)
                conn.execute(SCHEMA_VERSION_SQL)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now().isoformat()),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return

        if "schema_version" not in existing_tables:
            raise RuntimeError(
                "Database has no schema version; its existing schema is unknown. "
                "No tables or rows were changed."
            )

        version_row = conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if version_row is None:
            raise RuntimeError(
                "Database schema_version table is empty. No tables or rows were changed."
            )
        current_version = version_row["version"]
        if current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than "
                f"supported version {SCHEMA_VERSION}."
            )

        for target_version in range(current_version + 1, SCHEMA_VERSION + 1):
            migration = MIGRATIONS.get(target_version)
            if migration is None:
                raise RuntimeError(
                    f"No SQLite migration is available for schema version "
                    f"{target_version}."
                )
            foreign_keys_disabled_for_rebuild = target_version == 5
            try:
                if foreign_keys_disabled_for_rebuild:
                    enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
                    if enabled != 1:
                        raise RuntimeError(
                            "Cannot apply SQLite migration 5: foreign-key "
                            "enforcement is not enabled."
                        )
                    conn.execute("PRAGMA foreign_keys = OFF")
                    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 0:
                        raise RuntimeError(
                            "Cannot apply SQLite migration 5: foreign-key "
                            "enforcement could not be suspended for the table rebuild."
                        )
                conn.execute("BEGIN IMMEDIATE")
                migration(conn)
                conn.execute(
                    "UPDATE schema_version SET version = ?, applied_at = ?",
                    (target_version, datetime.now().isoformat()),
                )
                conn.commit()
                if foreign_keys_disabled_for_rebuild:
                    conn.execute("PRAGMA foreign_keys = ON")
                    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                        raise RuntimeError(
                            "SQLite migration 5 committed, but foreign-key "
                            "enforcement could not be restored."
                        )
                    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
                    if violations:
                        raise RuntimeError(
                            "SQLite migration 5 committed, but foreign-key "
                            "validation found violations."
                        )
            except Exception:
                conn.rollback()
                raise
            finally:
                if foreign_keys_disabled_for_rebuild:
                    if conn.in_transaction:
                        conn.rollback()
                    conn.execute("PRAGMA foreign_keys = ON")

        if current_version == SCHEMA_VERSION:
            try:
                conn.execute("BEGIN IMMEDIATE")
                _migration_4(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    finally:
        conn.close()

def reset_db(db_path: str = DB_PATH) -> None:
    """Drop and recreate all tables. For development databases only."""
    conn = get_db_connection(db_path)
    try:
        for table in (
            "transaction_messages",
            "identification_audit",
            "transactions",
            "bonds",
            "messages",
            "schema_version",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        conn.executescript(SCHEMA_VERSION_SQL)
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _sender_hash(sender_id: Any) -> Optional[str]:
    if sender_id is None or not str(sender_id).strip():
        return None
    salted_sender_id = f"{config.SENDER_HASH_SALT}:{sender_id}".encode("utf-8")
    return hashlib.sha256(salted_sender_id).hexdigest()


def _sender_name_for_storage(message: Dict[str, Any]) -> str:
    sender = str(message.get("sender") or "")
    sender_number = str(message.get("sender_number") or "")
    sender_digits = "".join(character for character in sender if character.isdigit())
    number_digits = "".join(
        character for character in sender_number if character.isdigit()
    )
    return "" if sender_digits and sender_digits == number_digits else sender


def insert_messages(
    conn: sqlite3.Connection,
    messages: List[Dict[str, Any]],
    upload_id: str,
) -> int:
    """Insert parsed messages once per stable message id."""
    inserted = 0
    for message in messages:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO messages (
                message_id, chat_id, sender, message_text, timestamp,
                chat_key, chat_name, chat_type, direction, is_own, is_media,
                media_kind, replying_to_text, noise_label, sender_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message["message_id"],
                upload_id,
                _sender_name_for_storage(message),
                message["message_text"],
                message["timestamp"],
                message.get("chat_key"),
                message.get("chat_name"),
                message.get("chat_type"),
                message.get("direction"),
                int(bool(message.get("is_own", False))),
                int(bool(message.get("is_media", False))),
                message.get("media_kind"),
                message.get("replying_to"),
                message.get("noise_label"),
                _sender_hash(message.get("sender_number")),
            ),
        )
        inserted += cursor.rowcount
    return inserted


def _context_identity(
    deal: Dict[str, Any],
    parsed_messages: List[Dict[str, Any]],
) -> str:
    chat_key = deal.get("chat_key")
    if not isinstance(chat_key, str) or not chat_key:
        raise ValueError("Cannot persist deal without its chat_key")

    message_ids = deal.get("message_ids")
    if not isinstance(message_ids, list) or not message_ids:
        message_number_map = deal.get("message_number_map")
        if isinstance(message_number_map, dict) and message_number_map:
            message_ids = [
                message_id
                for _, message_id in sorted(
                    message_number_map.items(),
                    key=lambda entry: int(entry[0][1:])
                    if isinstance(entry[0], str)
                    and entry[0].startswith("#")
                    and entry[0][1:].isdigit()
                    else float("inf"),
                )
            ]
        else:
            message_indices = deal.get("message_indices")
            if isinstance(message_indices, list):
                message_ids = [
                    parsed_messages[index]["message_id"]
                    for index in message_indices
                    if isinstance(index, int)
                    and 0 <= index < len(parsed_messages)
                    and parsed_messages[index].get("message_id")
                ]

    if not message_ids or not isinstance(message_ids[0], str) or not message_ids[0]:
        raise ValueError("Cannot persist deal without its first context message_id")

    context_key = json.dumps(
        [chat_key, message_ids[0]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(context_key.encode("utf-8")).hexdigest()


def _deterministic_transaction_id(
    source_hash: Optional[str],
    context_id: str,
    deal_index: int,
) -> str:
    key = json.dumps(
        [source_hash, context_id, deal_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def persist_upload(
    db_path: str,
    chat_id: str,
    parsed_messages: List[Dict[str, Any]],
    validated_deals: List[Dict[str, Any]],
    context_message_maps: Optional[Dict[str, Dict[str, str]]] = None,
    prompt_version: Optional[str] = None,
    model_name: Optional[str] = None,
    source_hash: Optional[str] = None,
    update_existing: bool = False,
) -> Dict[str, Any]:
    """
    Persist all messages, distinct bonds, transactions, and deal/message links.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    messages_stored = 0
    messages_already_present = 0
    bonds_stored = 0
    transactions_stored = 0
    transactions_updated = 0
    transactions_skipped = 0

    try:
        conn.execute("BEGIN TRANSACTION;")
        valid_deals = [
            deal for deal in validated_deals if not deal.get("extraction_failed")
        ]
        if source_hash is not None:
            existing_source_rows = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()[0]
            if existing_source_rows and not update_existing:
                existing_message_ids = {
                    row["message_id"]
                    for message in parsed_messages
                    for row in conn.execute(
                        "SELECT message_id FROM messages WHERE message_id = ?",
                        (message.get("message_id"),),
                    )
                }
                messages_already_present = len(existing_message_ids)
                conn.commit()
                return {
                    "already_stored": True,
                    "messages_inserted": 0,
                    "messages_already_present": messages_already_present,
                    "messages_skipped": len(parsed_messages) - messages_already_present,
                    "messages_stored": 0,
                    "bonds_stored": 0,
                    "transactions_inserted": 0,
                    "transactions_updated": 0,
                    "transactions_skipped": len(valid_deals),
                    "transactions_stored": 0,
                }

        existing_message_ids = {
            row["message_id"]
            for message in parsed_messages
            for row in conn.execute(
                "SELECT message_id FROM messages WHERE message_id = ?",
                (message.get("message_id"),),
            )
        }
        messages_stored = insert_messages(conn, parsed_messages, chat_id)
        messages_already_present = len(existing_message_ids)

        # TODO: Replace upload-time bond identity resolution with explicit match persistence.
        now_iso = datetime.now().isoformat()
        context_deal_indexes: Dict[str, int] = {}

        for deal in validated_deals:
            if deal.get("extraction_failed"):
                continue

            stable_context_id = _context_identity(deal, parsed_messages)
            deal_idx = context_deal_indexes.get(stable_context_id, 0)
            context_deal_indexes[stable_context_id] = deal_idx + 1
            tx_id = _deterministic_transaction_id(
                source_hash,
                stable_context_id,
                deal_idx,
            )
            existing_transaction = conn.execute(
                "SELECT transaction_id FROM transactions WHERE transaction_id = ?",
                (tx_id,),
            ).fetchone()
            if existing_transaction and not update_existing:
                transactions_skipped += 1
                continue

            bond_identity = (
                deal.get("bond_identity")
                or deal.get("bond_identity_text")
                or ""
            ).strip() or None
            isin = (
                deal.get("isin") or deal.get("isin_text") or ""
            ).strip().upper() or None
            identity_type = deal.get("identity_type") or (
                "ISIN" if isin else ("NAME_ONLY" if bond_identity else "UNKNOWN")
            )
            if identity_type not in ('ISIN', 'CUSIP', 'SERIES', 'NAME_ONLY', 'UNKNOWN'):
                identity_type = "UNKNOWN"

            # Check if bond already exists
            existing_bond = None
            if isin:
                cur = conn.execute("SELECT bond_id FROM bonds WHERE isin = ?", (isin,))
                row = cur.fetchone()
                if row:
                    existing_bond = row["bond_id"]

            if not existing_bond and bond_identity:
                cur = conn.execute("SELECT bond_id FROM bonds WHERE bond_identity = ?", (bond_identity,))
                row = cur.fetchone()
                if row:
                    existing_bond = row["bond_id"]

            if existing_bond:
                bond_id = existing_bond
            elif bond_identity:
                bond_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO bonds (
                        bond_id, isin, bond_identity, identity_type,
                        issuer_name, credit_rating, maturity_date,
                        coupon_rate, face_value, currency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bond_id,
                        isin,
                        bond_identity,
                        identity_type,
                        deal.get("issuer_name") or deal.get("issuer_text"),
                        deal.get("credit_rating"),
                        deal.get("maturity_date"),
                        float(deal["coupon_rate"]) if deal.get("coupon_rate") is not None else None,
                        float(deal["face_value"]) if deal.get("face_value") is not None else None,
                        deal.get("currency") or "INR",
                    ),
                )
                bonds_stored += 1
            else:
                bond_id = None

            # Transaction identity is based on this upload and its context position.
            normalized_quantity = deal.get("normalized_quantity")
            if not isinstance(normalized_quantity, dict):
                normalized_quantity = {}
            amount_inr = normalized_quantity.get("amount_inr")
            if isinstance(amount_inr, (dict, list)):
                amount_inr = json.dumps(amount_inr, ensure_ascii=False)
            quantity_value = normalized_quantity.get("value")
            if isinstance(quantity_value, (dict, list)):
                quantity_value = None
            settlement_date = deal.get("settlement_date")
            settlement_label = deal.get("settlement_label")
            settlement_ref = None
            if settlement_date is not None:
                settlement_ref = (
                    settlement_label
                    if settlement_label in ("T0", "T1", "T2")
                    else "OTHER"
                )
            status = deal.get("status")
            if status not in {
                "INITIATED",
                "NEGOTIATING",
                "CONCLUDED",
                "CANCELLED",
                "UNCERTAIN",
                "DISCARDED",
            }:
                status = "UNCERTAIN"
            validated_ytm = deal.get("ytm")
            if isinstance(validated_ytm, str):
                try:
                    validated_ytm = float(validated_ytm)
                except ValueError:
                    pass
            evidence = deal.get("evidence")
            warnings = deal.get("warnings")
            missing_fields = deal.get("missing_fields")

            transaction_columns = [
                "transaction_id", "source_hash", "bond_id", "buyer", "seller",
                "quantity", "quantity_unit", "is_indicative", "ytm",
                "final_price", "settlement_reference", "settlement_date",
                "status", "created_at", "updated_at", "context_id", "deal_index",
                "chat_key", "owner_side", "buyer_text", "seller_text",
                "quantity_text", "amount_inr", "units_assumed", "ytm_is_two_way",
                "settlement_text", "maturity_text", "maturity_ym",
                "partial_isin_text", "rfq_id", "package_note", "evidence_json",
                "warnings_json", "missing_fields_json", "confidence",
                "needs_review", "prompt_version", "model_name",
            ]
            transaction_values = (
                    tx_id,
                    source_hash,
                    bond_id,
                    deal.get("buyer") or deal.get("buyer_text"),
                    deal.get("seller") or deal.get("seller_text"),
                    float(quantity_value) if quantity_value is not None else None,
                    normalized_quantity.get("unit"),
                    (
                        int(bool(deal["is_indicative"]))
                        if deal.get("is_indicative") is not None
                        else None
                    ),
                    validated_ytm,
                    (
                        float(deal["price"])
                        if deal.get("price") is not None
                        else deal.get("final_price")
                    ),
                    settlement_ref,
                    settlement_date,
                    status,
                    deal.get("created_at") or now_iso,
                    now_iso,
                    stable_context_id,
                    deal_idx,
                    deal.get("chat_key"),
                    deal.get("owner_side"),
                    deal.get("buyer_text"),
                    deal.get("seller_text"),
                    deal.get("quantity_text"),
                    amount_inr,
                    (
                        int(bool(normalized_quantity["units_assumed"]))
                        if normalized_quantity.get("units_assumed") is not None
                        else None
                    ),
                    (
                        int(bool(deal["ytm_is_two_way"]))
                        if deal.get("ytm_is_two_way") is not None
                        else None
                    ),
                    deal.get("settlement_text"),
                    deal.get("maturity_text"),
                    deal.get("maturity_ym"),
                    deal.get("partial_isin_text"),
                    deal.get("rfq_id"),
                    deal.get("package_note"),
                    (
                        json.dumps(evidence, ensure_ascii=False)
                        if evidence is not None
                        else None
                    ),
                    (
                        json.dumps(warnings, ensure_ascii=False)
                        if warnings is not None
                        else None
                    ),
                    (
                        json.dumps(missing_fields, ensure_ascii=False)
                        if missing_fields is not None
                        else None
                    ),
                    deal.get("confidence"),
                    (
                        int(bool(deal["needs_review"]))
                        if deal.get("needs_review") is not None
                        else None
                    ),
                    deal.get("prompt_version") or prompt_version,
                    deal.get("model_name") or model_name,
            )
            if existing_transaction:
                update_columns = [
                    column
                    for column in transaction_columns
                    if column
                    not in {
                        "transaction_id",
                        "source_hash",
                        "created_at",
                        "context_id",
                        "deal_index",
                        "needs_review",
                    }
                ]
                conn.execute(
                    "UPDATE transactions SET "
                    + ", ".join(f"{column} = ?" for column in update_columns)
                    + " WHERE transaction_id = ?",
                    tuple(
                        transaction_values[transaction_columns.index(column)]
                        for column in update_columns
                    )
                    + (tx_id,),
                )
                transactions_updated += 1
            else:
                conn.execute(
                    "INSERT INTO transactions ("
                    + ", ".join(transaction_columns)
                    + ") VALUES ("
                    + ", ".join("?" for _ in transaction_columns)
                    + ")",
                    transaction_values,
                )
                transactions_stored += 1
            context_id = deal.get("context_id")
            message_number_map = (context_message_maps or {}).get(
                context_id,
            ) or deal.get("message_number_map") or {}
            cited_numbers = {
                line_number
                for field_lines in (evidence or {}).values()
                if isinstance(field_lines, list)
                for citation in field_lines
                if isinstance(citation, str)
                and (line_number := citation.strip()) in message_number_map
            }
            linked_message_ids = {
                message_number_map[number]
                for number in cited_numbers
                if message_number_map.get(number)
            }
            if not linked_message_ids:
                linked_message_ids.update(
                    message_number_map.values()
                )
            if not linked_message_ids:
                linked_message_ids.update(
                    parsed_messages[index]["message_id"]
                    for index in deal.get("message_indices", [])
                    if isinstance(index, int)
                    and 0 <= index < len(parsed_messages)
                )
            conn.executemany(
                """
                INSERT OR IGNORE INTO transaction_messages (
                    transaction_id, message_id
                ) VALUES (?, ?)
                """,
                [(tx_id, message_id) for message_id in linked_message_ids],
            )

        conn.commit()
        return {
            "already_stored": False,
            "messages_inserted": messages_stored,
            "messages_already_present": messages_already_present,
            "messages_skipped": max(
                0,
                len(parsed_messages) - messages_stored - messages_already_present,
            ),
            "messages_stored": messages_stored,
            "bonds_stored": bonds_stored,
            "transactions_inserted": transactions_stored,
            "transactions_updated": transactions_updated,
            "transactions_skipped": transactions_skipped,
            "transactions_stored": transactions_stored + transactions_updated,
        }

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _reference_record_payload(record: Any) -> Dict[str, Any]:
    return {
        "normalized_isin": record.normalized_isin,
        "normalized_issuer_key": record.normalized_issuer_key,
        "maturity_date": (
            record.maturity_date.isoformat()
            if record.maturity_date is not None
            else None
        ),
        "source_values": dict(record.source_values),
    }


def _match_evidence_payload(result: BondMatchResult) -> Dict[str, Any]:
    return {
        "source_evidence": [
            dict(vars(evidence)) for evidence in result.source_evidence
        ],
        "isin_source_evidence": [
            dict(vars(evidence)) for evidence in result.isin_source_evidence
        ],
    }


def _link_authorized_by_evidence(
    result: BondMatchResult,
    reference: Any,
) -> bool:
    if result.status == BondMatchStatus.PRIMARY_ISIN_MATCH:
        return (
            result.matching_method == "primary_isin"
            and result.isin_status == IsinStatus.VALID_ISIN
            and result.normalized_isin == reference.normalized_isin
            and bool(result.isin_source_evidence)
            and all(
                evidence.start >= 0
                and evidence.end > evidence.start
                and evidence.end <= len(evidence.source_message_text)
                and evidence.source_message_text[evidence.start:evidence.end]
                == evidence.matched_text
                and normalize_isin_candidate(evidence.matched_text)
                == reference.normalized_isin
                and (
                    (
                        evidence.relationship == "direct_message"
                        and evidence.message_id
                        and evidence.linked_to_message_id is None
                    )
                    or (
                        evidence.relationship == "linked_message"
                        and evidence.linked_to_message_id
                        and bool(evidence.link_evidence)
                    )
                )
                for evidence in result.isin_source_evidence
            )
        )
    if result.status == BondMatchStatus.SECONDARY_ISSUER_MATCH:
        return (
            result.matching_method == "secondary_issuer"
            and bool(result.raw_issuer_text and result.raw_issuer_text.strip())
            and bool(result.source_evidence)
            and any(
                evidence.field_name == "issuer"
                and evidence.start >= 0
                and evidence.end > evidence.start
                and evidence.end <= len(evidence.source_message_text)
                and normalize_issuer_key(evidence.source_value)
                == normalize_issuer_key(result.raw_issuer_text)
                and (
                    (
                        evidence.linked_to_message_id is None
                        and bool(evidence.message_id)
                        and evidence.link_evidence is None
                    )
                    or (
                        bool(evidence.linked_to_message_id)
                        and bool(evidence.link_evidence)
                    )
                )
                for evidence in result.source_evidence
            )
        )
    return False


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _insert_identification_audit(
    conn: sqlite3.Connection,
    audit_values: Tuple[Any, ...],
) -> int:
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO identification_audit (
            audit_id, transaction_id, identification_status, matching_method,
            reason, normalized_isin, raw_issuer_text, normalized_issuer_key,
            evidence_json, candidates_json, decision_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        audit_values,
    )
    return cursor.rowcount


def persist_identification_result(
    db_path: str,
    transaction_id: str,
    result: BondMatchResult,
) -> Dict[str, Any]:
    """Persist one Phase 4 decision atomically without inventing a bond."""
    if not isinstance(result, BondMatchResult):
        raise TypeError("result must be a BondMatchResult.")

    authorized_statuses = {
        BondMatchStatus.PRIMARY_ISIN_MATCH,
        BondMatchStatus.SECONDARY_ISSUER_MATCH,
    }
    authorized = result.status in authorized_statuses
    matched_reference = result.matched_reference
    if authorized and matched_reference is None:
        raise ValueError(
            "A uniquely matched result must include its matched reference record."
        )
    if authorized and not _link_authorized_by_evidence(result, matched_reference):
        raise ValueError(
            "The unique-match status is not supported by the required source evidence."
        )

    evidence_json = json.dumps(
        _match_evidence_payload(result),
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )
    candidates_json = json.dumps(
        [_reference_record_payload(record) for record in result.candidate_references],
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )
    decision_payload = {
        "status": result.status.value,
        "matching_method": result.matching_method,
        "reason": result.explanation,
        "normalized_isin": result.normalized_isin,
        "raw_issuer_text": result.raw_issuer_text,
        "normalized_issuer_key": result.normalized_issuer_key,
        "evidence": json.loads(evidence_json),
        "candidates": json.loads(candidates_json),
        "matched_reference": (
            _reference_record_payload(matched_reference)
            if matched_reference is not None
            else None
        ),
    }
    decision_key = hashlib.sha256(
        json.dumps(
            decision_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()

    conn = get_db_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        transaction = conn.execute(
            "SELECT bond_id FROM transactions WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        if transaction is None:
            raise ValueError(f"Transaction {transaction_id!r} does not exist.")

        current_bond_id = transaction["bond_id"]
        target_bond_id = current_bond_id
        if authorized:
            normalized_isin = matched_reference.normalized_isin
            if not normalized_isin:
                raise ValueError(
                    "A uniquely matched reference must have a normalized ISIN."
                )
            bond = conn.execute(
                "SELECT bond_id FROM bonds WHERE isin = ?",
                (normalized_isin,),
            ).fetchone()
            if bond is None:
                source_values = dict(matched_reference.source_values)
                issuer = source_values.get("NAME_OF_ISSUER", "").strip()
                description = source_values.get("SECURITY_DESCRIPTION", "").strip()
                maturity = matched_reference.maturity_date
                rate_text = source_values.get("INTEREST_RATE", "").strip()
                if rate_text:
                    try:
                        coupon_rate = float(rate_text.removesuffix("%").strip())
                    except ValueError as error:
                        raise ValueError(
                            "The matched reference has an invalid interest rate."
                        ) from error
                else:
                    coupon_rate = None
                target_bond_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO bonds (
                        bond_id, isin, bond_identity, identity_type, issuer_name,
                        maturity_date, coupon_rate, currency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_bond_id,
                        normalized_isin,
                        issuer or description or normalized_isin,
                        "ISIN",
                        issuer or None,
                        maturity.isoformat() if maturity is not None else None,
                        coupon_rate,
                        source_values.get("CURRENCY") or None,
                    ),
                )
            else:
                target_bond_id = bond["bond_id"]

            if current_bond_id and current_bond_id != target_bond_id:
                current_bond = conn.execute(
                    """
                    SELECT isin, identity_type
                    FROM bonds
                    WHERE bond_id = ?
                    """,
                    (current_bond_id,),
                ).fetchone()
                is_unresolved_placeholder = (
                    current_bond is not None
                    and current_bond["isin"] is None
                    and current_bond["identity_type"] == "UNKNOWN"
                )
                if not is_unresolved_placeholder:
                    raise ValueError(
                        "The transaction already has a different bond link; "
                        "the identification result was not applied."
                    )
        elif current_bond_id:
            current_bond = conn.execute(
                "SELECT isin, identity_type FROM bonds WHERE bond_id = ?",
                (current_bond_id,),
            ).fetchone()
            if (
                current_bond is not None
                and current_bond["isin"] is None
                and current_bond["identity_type"] == "UNKNOWN"
            ):
                target_bond_id = None

        conn.execute(
            """
            UPDATE transactions
            SET bond_id = ?, identification_status = ?
            WHERE transaction_id = ?
            """,
            (target_bond_id, result.status.value, transaction_id),
        )
        audit_inserted = _insert_identification_audit(
            conn,
            (
                str(uuid.uuid4()),
                transaction_id,
                result.status.value,
                result.matching_method,
                result.explanation,
                result.normalized_isin,
                result.raw_issuer_text,
                result.normalized_issuer_key,
                evidence_json,
                candidates_json,
                decision_key,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return {
            "transaction_id": transaction_id,
            "identification_status": result.status.value,
            "bond_id": target_bond_id,
            "bond_linked": bool(authorized),
            "audit_inserted": bool(audit_inserted),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_database_data(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Retrieve stored rows and transaction-to-message links."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        tx_rows = conn.execute(
            """
            SELECT t.*, b.bond_identity, b.isin, b.issuer_name, b.coupon_rate
            FROM transactions t
            LEFT JOIN bonds b ON t.bond_id = b.bond_id
            ORDER BY t.created_at DESC
            """
        ).fetchall()
        linked_message_rows = conn.execute(
            """
            SELECT transaction_id, message_id
            FROM transaction_messages
            ORDER BY transaction_id, message_id
            """
        ).fetchall()
        linked_message_ids: Dict[str, List[str]] = {}
        for row in linked_message_rows:
            linked_message_ids.setdefault(row["transaction_id"], []).append(
                row["message_id"]
            )
        transactions = []
        for row in tx_rows:
            transaction = dict(row)
            transaction["message_ids"] = linked_message_ids.get(
                transaction["transaction_id"],
                [],
            )
            transaction["settlement_display"] = (
                transaction.get("settlement_text")
                or transaction.get("settlement_date")
                or "—"
            )
            ytm = transaction.get("ytm")
            transaction["ytm_display"] = (
                (
                    f"{ytm:g}%"
                    if isinstance(ytm, (int, float))
                    else str(ytm)
                )
                if ytm is not None
                else "—"
            )
            transactions.append(transaction)

        bonds_rows = conn.execute("SELECT * FROM bonds ORDER BY bond_identity ASC").fetchall()
        msgs_rows = conn.execute("SELECT * FROM messages ORDER BY timestamp ASC LIMIT 500").fetchall()

        return {
            "transactions": transactions,
            "bonds": [dict(r) for r in bonds_rows],
            "messages": [dict(r) for r in msgs_rows],
            "counts": {
                "transactions": len(tx_rows),
                "bonds": len(bonds_rows),
                "messages": len(msgs_rows),
            },
        }
    finally:
        conn.close()
