"""
Database module for Bond Trade Extraction App.
Implements the exact SQLite schema required:
- messages
- bonds
- transactions
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = "bond_deals.db"

# Exact DDL specified in requirements
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,     -- UUID
    chat_id         TEXT NOT NULL,
    sender          TEXT NOT NULL,
    message_text    TEXT NOT NULL,
    timestamp       TEXT NOT NULL,         -- ISO 8601 datetime
    transaction_id  TEXT,                  -- nullable FK -> transactions.transaction_id
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
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
    settlement_reference          TEXT CHECK (settlement_reference IN ('T0','T1','T2')),
    settlement_date                 TEXT,       -- ISO date
    status                            TEXT NOT NULL CHECK (status IN
                                          ('INCOMPLETE','CONCLUDED','DISCARDED')),
    created_at                          TEXT NOT NULL,
    updated_at                           TEXT NOT NULL,
    FOREIGN KEY (bond_id) REFERENCES bonds (bond_id)
);
"""


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables with exact DDL."""
    with get_db_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def persist_upload(
    db_path: str,
    chat_id: str,
    parsed_messages: List[Dict[str, Any]],
    validated_deals: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Persist all messages, distinct bonds, and transactions in a single transaction.
    Links messages belonging to a trade window with the transaction_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    messages_stored = 0
    bonds_stored = 0
    transactions_stored = 0

    try:
        conn.execute("BEGIN TRANSACTION;")

        # 1. Map each deal to a created or existing bond
        # To avoid duplicate bond records for identical bonds, check by isin or bond_identity
        deal_to_tx_id = {}
        now_iso = datetime.now().isoformat()

        for deal_idx, deal in enumerate(validated_deals):
            if not deal.get("is_valid", True):
                # If discarded or invalid, don't store transaction
                if deal.get("status") not in ("CONCLUDED", "INCOMPLETE"):
                    continue

            bond_identity = (deal.get("bond_identity") or "UNKNOWN").strip()
            isin = (deal.get("isin") or "").strip().upper() or None
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
                        deal.get("issuer_name"),
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
            deal_to_tx_id[deal_idx] = tx_id

            settlement_ref = deal.get("settlement_reference") or "T1"
            if settlement_ref not in ('T0', 'T1', 'T2'):
                settlement_ref = "T1"

            status = deal.get("status") or "CONCLUDED"
            if status not in ('INCOMPLETE', 'CONCLUDED', 'DISCARDED'):
                status = "CONCLUDED"

            conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, bond_id, buyer, seller,
                    quantity, quantity_unit, is_indicative, ytm,
                    final_price, settlement_reference, settlement_date,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tx_id,
                    bond_id,
                    deal.get("buyer"),
                    deal.get("seller"),
                    float(deal["quantity"]) if deal.get("quantity") is not None else None,
                    deal.get("quantity_unit") or "Cr",
                    int(deal.get("is_indicative", 0)),
                    float(deal["ytm"]) if deal.get("ytm") is not None else None,
                    float(deal["final_price"]) if deal.get("final_price") is not None else None,
                    settlement_ref,
                    deal.get("settlement_date"),
                    status,
                    deal.get("created_at") or now_iso,
                    deal.get("updated_at") or now_iso,
                ),
            )
            transactions_stored += 1

        # 2. Insert all parsed messages and link transaction_id if part of a trade window
        # Create a mapping of message index -> transaction_id
        msg_tx_map = {}
        for deal_idx, deal in enumerate(validated_deals):
            tx_id = deal_to_tx_id.get(deal_idx)
            if tx_id and "message_indices" in deal:
                for idx in deal["message_indices"]:
                    msg_tx_map[idx] = tx_id

        for idx, msg in enumerate(parsed_messages):
            msg_id = msg.get("message_id") or str(uuid.uuid4())
            tx_id = msg_tx_map.get(idx) or msg.get("transaction_id")

            conn.execute(
                """
                INSERT INTO messages (
                    message_id, chat_id, sender, message_text, timestamp, transaction_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    chat_id,
                    msg["sender"],
                    msg["message_text"],
                    msg["timestamp"],
                    tx_id,
                ),
            )
            messages_stored += 1

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
    """Retrieve stored rows for all three tables."""
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

        bonds_rows = conn.execute("SELECT * FROM bonds ORDER BY bond_identity ASC").fetchall()
        msgs_rows = conn.execute("SELECT * FROM messages ORDER BY timestamp ASC LIMIT 500").fetchall()

        return {
            "transactions": [dict(r) for r in tx_rows],
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
