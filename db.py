"""
Database module for Bond Trade Extraction App.
Implements the exact SQLite schema required:
- messages
- bonds
- transactions
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config

DB_PATH = "bond_deals.db"
SCHEMA_VERSION = 2

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
    bond_id                TEXT NOT NULL,
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
    created_at                          TEXT NOT NULL,
    updated_at                           TEXT NOT NULL,
    context_id                          TEXT,
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
    FOREIGN KEY (bond_id) REFERENCES bonds (bond_id)
);

CREATE TABLE IF NOT EXISTS transaction_messages (
    transaction_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    PRIMARY KEY (transaction_id, message_id),
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id),
    FOREIGN KEY (message_id) REFERENCES messages (message_id)
);
"""

SCHEMA_VERSION_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER NOT NULL,
    applied_at   TEXT NOT NULL
);
"""


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the current schema, refusing implicit legacy migrations."""
    conn = get_db_connection(db_path)
    try:
        existing_tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        conn.executescript(SCHEMA_VERSION_SQL)
        version_row = conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if version_row is None and existing_tables:
            raise RuntimeError(
                "Database has no schema version and may use an older schema; "
                "run reset_db(db_path) to recreate it (development only)."
            )
        if version_row is not None and version_row["version"] < SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version_row['version']} is older than "
                f"required version {SCHEMA_VERSION}; run reset_db(db_path) to "
                "recreate it (development only)."
            )
        if version_row is not None and version_row["version"] > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version_row['version']} is newer than "
                f"supported version {SCHEMA_VERSION}."
            )
        conn.executescript(SCHEMA_SQL)
        if version_row is None:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )
        conn.commit()
    finally:
        conn.close()

def reset_db(db_path: str = DB_PATH) -> None:
    """Drop and recreate all tables. For development databases only."""
    conn = get_db_connection(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "transaction_messages",
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
                media_kind, replying_to_text, noise_label, sender_hash,
                chat_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                upload_id,
            ),
        )
        inserted += cursor.rowcount
    return inserted


def persist_upload(
    db_path: str,
    chat_id: str,
    parsed_messages: List[Dict[str, Any]],
    validated_deals: List[Dict[str, Any]],
    context_message_maps: Optional[Dict[str, Dict[str, str]]] = None,
    prompt_version: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Persist all messages, distinct bonds, transactions, and deal/message links.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    messages_stored = 0
    bonds_stored = 0
    transactions_stored = 0

    try:
        conn.execute("BEGIN TRANSACTION;")
        messages_stored = insert_messages(conn, parsed_messages, chat_id)

        # IDENTITY-LADDER: bond identity resolution is intentionally unchanged.
        now_iso = datetime.now().isoformat()

        for deal_idx, deal in enumerate(validated_deals):
            if deal.get("extraction_failed"):
                continue

            bond_identity = (
                deal.get("bond_identity")
                or deal.get("bond_identity_text")
                or "UNKNOWN"
            ).strip()
            isin = (
                deal.get("isin") or deal.get("isin_text") or ""
            ).strip().upper() or None
            identity_type = deal.get("identity_type") or ("ISIN" if isin else "NAME_ONLY")
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
            else:
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

            # Insert transaction
            tx_id = str(uuid.uuid4())
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
                    validated_ytm = None
            evidence = deal.get("evidence")
            warnings = deal.get("warnings")
            missing_fields = deal.get("missing_fields")

            conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, bond_id, buyer, seller,
                    quantity, quantity_unit, is_indicative, ytm,
                    final_price, settlement_reference, settlement_date,
                    status, created_at, updated_at, context_id, chat_key,
                    owner_side, buyer_text, seller_text, quantity_text,
                    amount_inr, units_assumed, ytm_is_two_way, settlement_text,
                    maturity_text, maturity_ym, partial_isin_text, rfq_id,
                    package_note, evidence_json, warnings_json,
                    missing_fields_json, confidence, needs_review,
                    prompt_version, model_name
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    tx_id,
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
                    float(validated_ytm) if validated_ytm is not None else None,
                    (
                        float(deal["price"])
                        if deal.get("price") is not None
                        else deal.get("final_price")
                    ),
                    settlement_ref,
                    settlement_date,
                    status,
                    deal.get("created_at") or now_iso,
                    deal.get("updated_at") or now_iso,
                    deal.get("context_id"),
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
                ),
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
            "messages_stored": messages_stored,
            "bonds_stored": bonds_stored,
            "transactions_stored": transactions_stored,
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
                f"{ytm:g}%" if ytm is not None else "—"
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
