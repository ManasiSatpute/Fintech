"""
WhatsApp -> Bond Deal Extraction -> SQL Pipeline
=================================================

Flow:
  1. Parse raw WhatsApp .txt export into structured messages.
  2. Drop noise deterministically (system msgs, greetings, chit-chat) using
     rule-based filters -- NO LLM call for this step (cheap + fast).
  3. Group remaining messages into "conversation windows" (a burst of
     messages between the same participants that looks like one deal).
  4. Only windows that pass a bond-relevance check are sent to the
     extraction agent (LLM) -- this is the "send only valid data to agent"
     requirement.
  5. Agent returns strict JSON: Bond, Buyer, Seller, Price, deal_time, deal_type.
  6. Validate the agent's output (business rules) before inserting.
  7. Idempotent upsert into SQLite (swap the connection string for any
     other SQL DB -- the schema/SQL is standard).

Only step 4 (call_extraction_agent) talks to an LLM. Everything else is
deterministic so you get a repeatable, auditable pipeline.
"""

import re
import json
import sqlite3
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional


# 1. PARSE RAW WHATSAPP EXPORT


# Matches: [11/08/26, 3:00:27 PM] Sender Name: message text
LINE_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}:\d{2}\s?[APap][Mm])\]\s*"
    r"(?P<sender>[^:]+):\s*(?P<message>.*)$"
)

SYSTEM_SENDERS = {"Chat"}  
SYSTEM_MSG_MARKERS = (
    "Messages and calls are end-to-end encrypted",
    "created group",
    "changed the subject",
    "changed this group's icon",
)


@dataclass
class Message:
    ts: datetime
    sender: str
    text: str


def parse_whatsapp_export(path: str) -> list[Message]:
    messages: list[Message] = []
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # WhatsApp exports can have multi-line messages; stitch continuation
    # lines (ones that don't match the timestamp pattern) onto the previous.
    for raw_line in raw.splitlines():
        line = raw_line.replace("\u200e", "").strip()  # strip LRM marks
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            ts = datetime.strptime(f"{m['date']} {m['time']}", "%d/%m/%y %I:%M:%S %p")
            messages.append(Message(ts=ts, sender=m["sender"].strip(), text=m["message"].strip()))
        elif messages:
            # continuation of previous message
            messages[-1].text += " " + line

    return messages



# 2. DETERMINISTIC NOISE FILTER


CHIT_CHAT_PATTERNS = [
    r"^hello", r"^hi\b", r"^hey", r"good\s?(morning|evening|afternoon)",
    r"how are you", r"had lunch", r"thanks", r"thank you", r"^ok$", r"^okay$",
    r"what is going on", r"all good", r"that's great",
]
CHIT_CHAT_RE = re.compile("|".join(CHIT_CHAT_PATTERNS), re.IGNORECASE)


def is_system_message(msg: Message) -> bool:
    if msg.sender in SYSTEM_SENDERS:
        return True
    return any(marker in msg.text for marker in SYSTEM_MSG_MARKERS)


def is_chit_chat(msg: Message) -> bool:
    return bool(CHIT_CHAT_RE.search(msg.text.strip()))


def clean_messages(messages: list[Message]) -> list[Message]:
    return [m for m in messages if not is_system_message(m) and not is_chit_chat(m)]


# 3. GROUP INTO CONVERSATION WINDOWS


WINDOW_GAP = timedelta(minutes=15)  # new window if silence exceeds this


def group_into_windows(messages: list[Message]) -> list[list[Message]]:
    windows: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages:
        if current and (msg.ts - current[-1].ts) > WINDOW_GAP:
            windows.append(current)
            current = []
        current.append(msg)
    if current:
        windows.append(current)
    return windows



# 4. BOND-RELEVANCE CHECK (the gate before hitting the agent)


# Cheap deterministic signals that a window is *likely* a bond deal.
BOND_KEYWORDS_RE = re.compile(
    r"\b(GS|SDL|T[- ]?bill|ISIN|coupon|yield|face value|cr|crore|lakh)\b", re.IGNORECASE
)
PRICE_RE = re.compile(r"\b\d{1,2}\.\d{1,3}\b")           # e.g. 6.65, 6.465
AMOUNT_RE = re.compile(r"\b\d+(\.\d+)?\s?(cr|crore|lakh)\b", re.IGNORECASE)
ACTION_RE = re.compile(r"\b(buy|sell|bid|offer|done)\b", re.IGNORECASE)


def is_bond_relevant(window: list[Message]) -> bool:
    blob = " ".join(m.text for m in window)
    signals = [
        bool(BOND_KEYWORDS_RE.search(blob)),
        bool(PRICE_RE.search(blob)),
        bool(AMOUNT_RE.search(blob)),
        bool(ACTION_RE.search(blob)),
    ]
    # require at least 3 of 4 signals -> reduces false positives sent to the agent
    return sum(signals) >= 3


def window_to_text(window: list[Message]) -> str:
    return "\n".join(f"[{m.ts.strftime('%H:%M:%S')}] {m.sender}: {m.text}" for m in window)


# --------------------------------------------------------------------------
# 5. AGENT CALL (only relevant windows reach this point)
#    Real implementation: LangChain `create_agent` + a local Ollama model.
#    The agent's ONLY job is structured extraction -- it never touches the
#    DB. Storage/validation stay deterministic in Python (steps 6-7 below),
#    same split you use in HexStrike between LLM synthesis and hard logic.
# --------------------------------------------------------------------------

from pydantic import BaseModel, Field
from langchain.agents import create_agent

EXTRACTION_SYSTEM_PROMPT = """You are a bond-deal extraction agent for a fintech desk.
You will be given a short WhatsApp conversation snippet that has already been
pre-filtered as likely containing a bond trade negotiation.

Extract the FINAL agreed deal only (ignore intermediate quotes that were rejected
or renegotiated). Never invent a bond name, price, or party that isn't grounded
in the text -- if a field is not present, leave it empty/null and set
is_deal_confirmed to false."""


class BondDeal(BaseModel):
    """Structured bond deal extracted from a WhatsApp conversation window."""
    bond: str = Field(description='Bond description, e.g. "6.65% GS 2026"')
    buyer: str = Field(description="Name of the party buying the bond")
    seller: str = Field(description="Name of the party selling the bond")
    price: float = Field(description="Final agreed price/yield")
    amount: str = Field(description='Final agreed deal size, e.g. "5 Cr"')
    deal_time: str = Field(description="Timestamp of the confirming message, HH:MM:SS")
    deal_type: str = Field(description='"buy" or "sell", from the buyer\'s perspective')
    is_deal_confirmed: bool = Field(description="True only if both sides explicitly confirmed")


# Build once and reuse across windows -- creating the agent per-call is wasteful.
# Model string follows LangChain's "provider:model" convention, e.g.
# "ollama:qwen2.5" or "ollama:minimax-m3" -- swap for whatever you have pulled.
def build_extraction_agent(model: str = "ollama:qwen2.5"):
    return create_agent(
        model=model,
        tools=[],  # pure extraction -- no tools needed, keeps output deterministic
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        response_format=BondDeal,  # forces structured output, no JSON-parsing fragility
    )


_AGENT = None  # lazy singleton so unit tests / mock mode don't require Ollama running


def call_extraction_agent(conversation_text: str, model: str = "ollama:qwen2.5",
                           use_mock: bool = False) -> dict:
    """
    Turns a pre-filtered conversation window into structured JSON via a local
    Ollama model. Requires `ollama serve` running and the model pulled
    (e.g. `ollama pull qwen2.5`).

    use_mock=True bypasses the LLM entirely (rule-based stand-in) so you can
    run/test the surrounding pipeline without Ollama installed.
    """
    if use_mock:
        return json.loads(_mock_agent(conversation_text))

    global _AGENT
    if _AGENT is None:
        _AGENT = build_extraction_agent(model=model)

    result = _AGENT.invoke({"messages": [{"role": "user", "content": conversation_text}]})
    print(result)
    deal: BondDeal = result["structured_response"]
    return deal.model_dump()


def _mock_agent(conversation_text: str) -> str:
    """Rule-based stand-in for the LLM call, used only for local testing."""
    lines = conversation_text.splitlines()
    speakers = []
    bond, price, amount, deal_time = None, None, None, None
    buyer = seller = None
    confirmed = False

    for line in lines:
        m = re.match(r"\[(?P<t>[\d:]+)\]\s*(?P<who>[^:]+):\s*(?P<msg>.*)", line)
        if not m:
            continue
        who, msg, t = m["who"].strip(), m["msg"].strip(), m["t"]
        if who not in speakers:
            speakers.append(who)

        bm = re.search(r"(\d+(\.\d+)?\s?(Cr|Crore))?\s*(\d{1,2}\.\d{1,2}\s?(GS|SDL)\s?\d{4})", msg, re.IGNORECASE)
        if bm:
            bond = bm.group(4)
        # a bare "buy" declaration (not the "buy or sell" question) marks the buyer
        if re.match(r"^\s*buy\s*$", msg, re.IGNORECASE) and buyer is None:
            buyer = who
        pm = re.search(r"\b\d{1,2}\.\d{1,3}\b", msg)
        if pm:
            price = float(pm.group())
        am = re.search(r"\b(\d+(\.\d+)?)\s?(cr|crore)\b", msg, re.IGNORECASE)
        if am:
            amount = f"{am.group(1)} Cr"
        if re.search(r"\bdone\b", msg, re.IGNORECASE):
            confirmed = True
            deal_time = t

    if buyer and len(speakers) == 2:
        seller = [s for s in speakers if s != buyer][0]

    result = {
        "bond": bond or "UNKNOWN",
        "buyer": buyer or "UNKNOWN",
        "seller": seller or "UNKNOWN",
        "price": price,
        "amount": amount or "UNKNOWN",
        "deal_time": deal_time,
        "deal_type": "buy",
        "is_deal_confirmed": confirmed,
    }
    return json.dumps(result)



# 6. VALIDATION (business rules before touching the DB)


def validate_deal(deal: dict) -> tuple[bool, str]:
    if not deal.get("is_deal_confirmed"):
        return False, "deal not confirmed in conversation"
    if deal.get("buyer") in (None, "UNKNOWN") or deal.get("seller") in (None, "UNKNOWN"):
        return False, "missing buyer/seller"
    if deal["buyer"] == deal["seller"]:
        return False, "buyer and seller are the same party"
    if not isinstance(deal.get("price"), (int, float)) or deal["price"] <= 0:
        return False, "invalid price"
    if deal.get("bond", "UNKNOWN") == "UNKNOWN":
        return False, "bond not identified"
    return True, "ok"



# 7. SQL STORAGE (SQLite here; same DDL/DML style works on Postgres/MySQL)


DDL = """
CREATE TABLE IF NOT EXISTS bond_deals (
    deal_id      TEXT PRIMARY KEY,     -- hash of natural key, for idempotency
    bond         TEXT NOT NULL,
    buyer        TEXT NOT NULL,
    seller       TEXT NOT NULL,
    price        REAL NOT NULL,
    amount       TEXT,
    deal_time    TEXT NOT NULL,
    deal_type    TEXT NOT NULL CHECK (deal_type IN ('buy','sell')),
    source_text  TEXT,                 -- raw window, for audit trail
    inserted_at  TEXT DEFAULT (datetime('now'))
);
"""

UPSERT = """
INSERT INTO bond_deals (deal_id, bond, buyer, seller, price, amount, deal_time, deal_type, source_text)
VALUES (:deal_id, :bond, :buyer, :seller, :price, :amount, :deal_time, :deal_type, :source_text)
ON CONFLICT(deal_id) DO UPDATE SET
    price = excluded.price,
    amount = excluded.amount,
    deal_time = excluded.deal_time,
    source_text = excluded.source_text;
"""


def make_deal_id(deal: dict) -> str:
    # natural key: same bond + parties + day -> same deal (re-processing safe)
    day = (deal.get("deal_time") or "")[:10]
    key = f"{deal['bond']}|{deal['buyer']}|{deal['seller']}|{day}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def store_deal(db_path: str, deal: dict, source_text: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(DDL)
        conn.execute(UPSERT, {
            "deal_id": make_deal_id(deal),
            "bond": deal["bond"],
            "buyer": deal["buyer"],
            "seller": deal["seller"],
            "price": deal["price"],
            "amount": deal.get("amount"),
            "deal_time": deal.get("deal_time"),
            "deal_type": deal.get("deal_type", "buy"),
            "source_text": source_text,
        })
        conn.commit()
    finally:
        conn.close()



# 8. MAIN PIPELINE


def run_pipeline(chat_path: str, db_path: str = "bond_deals.db",
                  model: str = "gemma4:31b-cloud", use_mock: bool = False):
    messages = parse_whatsapp_export(chat_path)
    cleaned = clean_messages(messages)
    windows = group_into_windows(cleaned)

    stored, skipped = 0, 0
    for window in windows:
        if not is_bond_relevant(window):
            continue  # never leaves this process -> agent is never called

        text = window_to_text(window)
        deal = call_extraction_agent(text, model=model, use_mock=use_mock)  # <- only valid data reaches here
        ok, reason = validate_deal(deal)

        if ok:
            store_deal(db_path, deal, source_text=text)
            stored += 1
            print(f"[STORED] {deal['bond']} | {deal['buyer']} <- {deal['seller']} "
                  f"@ {deal['price']} ({deal['amount']})")
        else:
            skipped += 1
            print(f"[SKIPPED] reason='{reason}' | payload={deal}")

    print(f"\nDone. {stored} deal(s) stored, {skipped} window(s) skipped.")


if __name__ == "__main__":
    # Set use_mock=False once Ollama is running locally (`ollama serve`) and
    # the model in `model=` has been pulled (`ollama pull qwen2.5`).
    run_pipeline(
        "/Users/manasi/Downloads/Whatsaapp_to_Sql/chat_2.txt",
        db_path="Whatsaapp_to_Sql\\bond_deals.db",
        model="ollama:gemma4:cloud",
        use_mock=True,
    )