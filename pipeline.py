"""
Pipeline module for WhatsApp -> Bond Trade Extraction.
Handles:
1. Multi-format parsing (plain-text WhatsApp exports & JSON/JSONL)
2. Deterministic noise cleaning & routing
3. Grouping into trade contexts (15 min gap threshold)
4. LLM extraction via langchain-ollama using cloud model (gemma4:31b-cloud)
5. Token usage aggregation & context window query
6. Business-rule validation
"""

import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ollama
from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:31b-cloud"

# --------------------------------------------------------------------------
# 1. PARSING RAW CHAT EXPORTS (TXT & JSON)
# --------------------------------------------------------------------------

# iOS format: [11/08/26, 3:00:27 PM] Sender: Message
LINE_RE_IOS = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s*(?P<sender>[^:]+):\s*(?P<msg>.*)$"
)

# Android format: 11/08/2026, 15:00 - Sender: Message or 11/08/26, 3:00 pm - Sender: Message
LINE_RE_ANDROID = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s*-\s*(?P<sender>[^:]+):\s*(?P<msg>.*)$"
)

TIMESTAMP_FORMATS = [
    "%d/%m/%y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%m/%d/%y %I:%M:%S %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%y %I:%M %p",
    "%d/%m/%Y %I:%M %p",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%d/%m/%Y %H:%M",
]


def parse_timestamp_str(date_str: str, time_str: str) -> datetime:
    combined = f"{date_str.strip()} {time_str.strip()}"
    # Standardize spaces and AM/PM
    combined = re.sub(r"\s+", " ", combined)
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    # Fallback to current datetime
    return datetime.now()


def parse_raw_chat_content(raw_content: str, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parses either WhatsApp text format or JSON/JSONL format into uniform message dicts:
    {
        "message_id": UUID,
        "chat_id": str,
        "sender": str,
        "message_text": str,
        "timestamp": ISO 8601 string,
        "dt": datetime object,
        "group_name": Optional[str],
        "channel": Optional[str]
    }
    """
    if not chat_id:
        chat_id = str(uuid.uuid4())

    # Clean null bytes (from UTF-16) and unicode directional marks
    content = raw_content.replace("\x00", "").replace("\u200e", "").replace("\u202f", " ")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines:
        return []

    # Check if first line is JSON
    is_json = False
    try:
        first_parsed = json.loads(lines[0])
        if isinstance(first_parsed, dict) and ("message" in first_parsed or "message_text" in first_parsed or "sender_name" in first_parsed):
            is_json = True
    except Exception:
        is_json = False

    messages: List[Dict[str, Any]] = []

    if is_json:
        for idx, line in enumerate(lines):
            try:
                item = json.loads(line)
            except Exception:
                continue

            raw_ts = item.get("timestamp") or item.get("ts") or datetime.now().isoformat()
            dt = None
            if isinstance(raw_ts, str):
                # Try parsing "08/07/2026, 10:49:59 AM" or similar
                ts_clean = raw_ts.replace(",", "")
                for fmt in [
                    "%d/%m/%Y %I:%M:%S %p",
                    "%m/%d/%Y %I:%M:%S %p",
                    "%d/%m/%y %I:%M:%S %p",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                ]:
                    try:
                        dt = datetime.strptime(ts_clean, fmt)
                        break
                    except ValueError:
                        continue
            if dt is None:
                dt = datetime.now()

            sender = item.get("sender_name") or item.get("sender") or item.get("sender_number") or "Unknown"
            msg_text = item.get("message") or item.get("message_text") or ""
            group_name = item.get("group_name") or item.get("chat_type") or "Direct"

            messages.append({
                "message_id": str(uuid.uuid4()),
                "chat_id": chat_id,
                "sender": sender.strip(),
                "message_text": msg_text.strip(),
                "timestamp": dt.isoformat(),
                "dt": dt,
                "group_name": group_name,
                "channel": group_name,
            })
    else:
        # Standard WhatsApp export
        for line in lines:
            m_ios = LINE_RE_IOS.match(line)
            m_android = LINE_RE_ANDROID.match(line)
            m = m_ios or m_android

            if m:
                dt = parse_timestamp_str(m.group("date"), m.group("time"))
                sender = m.group("sender").strip()
                text = m.group("msg").strip()
                messages.append({
                    "message_id": str(uuid.uuid4()),
                    "chat_id": chat_id,
                    "sender": sender,
                    "message_text": text,
                    "timestamp": dt.isoformat(),
                    "dt": dt,
                    "group_name": "Default",
                    "channel": "Default",
                })
            elif messages:
                # Continuation of multi-line message
                messages[-1]["message_text"] += "\n" + line

    return messages


# --------------------------------------------------------------------------
# 2. NOISE FILTER & ROUTING
# --------------------------------------------------------------------------

SYSTEM_SENDERS = {"Chat", "System"}
SYSTEM_MARKERS = [
    "messages and calls are end-to-end encrypted",
    "created group",
    "changed the subject",
    "changed this group's icon",
    "left",
    "added",
    "security code changed",
    "[media message]",
    "[image message]",
]

CHIT_CHAT_PATTERNS = [
    r"^hello\b", r"^hi\b", r"^hey\b", r"^good\s?(morning|evening|afternoon)",
    r"how are you", r"had lunch", r"thanks", r"thank you", r"^ok$", r"^okay$",
    r"^all good", r"^that's great", r"what is going on", r"see you", r"^nice!?$",
    r"^working on it", r"cheating mat kar", r"500 gpay", r"lost 3 deals",
    r"unproductive work", r"oil surges",
]
CHIT_CHAT_RE = re.compile("|".join(CHIT_CHAT_PATTERNS), re.IGNORECASE)

BOND_KEYWORDS_RE = re.compile(
    r"\b(GS|SDL|T[- ]?bill|ISIN|INE[A-Z0-9]{9}|Oxyzo|Sammaan|Muthoot|Vedika|NHAI|HUDCO|Hinduja|Kerala|Tax[- ]?Free|coupon|yield|ytm|face value|cr|crore|lakh|lacs|units|offer|bid|sold|bought|done|closed)\b",
    re.IGNORECASE
)
PRICE_OR_RATE_RE = re.compile(r"\b\d{1,2}\.\d{1,4}%?\b")
AMOUNT_RE = re.compile(r"\b\d+(\.\d+)?\s?(cr|crore|lakh|lacs|units)\b", re.IGNORECASE)
ACTION_RE = re.compile(r"\b(buy|sell|bid|offer|done|close|closed|confirm|confirmed|holding)\b", re.IGNORECASE)


def is_system_or_media(msg: Dict[str, Any]) -> bool:
    sender = msg["sender"]
    text = msg["message_text"].lower().strip()
    if sender in SYSTEM_SENDERS:
        return True
    if any(marker in text for marker in SYSTEM_MARKERS):
        return True
    return False


def is_chit_chat(msg: Dict[str, Any]) -> bool:
    text = msg["message_text"].strip()
    # Don't drop messages that explicitly contain bond identifiers or ISINs
    if re.search(r"INE[A-Z0-9]{9}", text, re.IGNORECASE):
        return False
    if re.search(r"\b\d+(\.\d+)?\s?(cr|crore|lakh|units)\b", text, re.IGNORECASE) and re.search(r"\b\d{1,2}\.\d{1,3}\b", text):
        return False
    return bool(CHIT_CHAT_RE.search(text))


# --------------------------------------------------------------------------
# 3. GROUPING INTO TRADE CONTEXTS
# --------------------------------------------------------------------------

WINDOW_GAP = timedelta(minutes=15)


def is_bond_trade_relevant(window: List[Dict[str, Any]]) -> bool:
    blob = " ".join(m["message_text"] for m in window)
    has_bond_key = bool(BOND_KEYWORDS_RE.search(blob))
    has_price = bool(PRICE_OR_RATE_RE.search(blob))
    has_amount = bool(AMOUNT_RE.search(blob))
    has_action = bool(ACTION_RE.search(blob))

    # High signal: Contains ISIN
    if re.search(r"INE[A-Z0-9]{9}", blob, re.IGNORECASE):
        return True

    # High signal: Bond keyword + price + action/amount
    signals = [has_bond_key, has_price, has_amount, has_action]
    return sum(signals) >= 3


def group_into_trade_contexts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups clean messages into per-negotiation trade contexts.
    Returns list of trade context dicts:
    {
        "context_id": str,
        "group_name": str,
        "messages": [...],
        "message_indices": [...],
        "snippet_text": str
    }
    """
    # Separate by channel/group first if multiple channels exist
    channels: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, msg in enumerate(messages):
        ch = msg.get("group_name") or "Default"
        channels.setdefault(ch, []).append((idx, msg))

    contexts: List[Dict[str, Any]] = []

    for ch_name, ch_items in channels.items():
        current_burst: List[Tuple[int, Dict[str, Any]]] = []

        for idx, msg in ch_items:
            # Check noise
            if is_system_or_media(msg):
                continue

            if current_burst:
                prev_msg = current_burst[-1][1]
                gap = msg["dt"] - prev_msg["dt"]
                if gap > WINDOW_GAP:
                    # Check if previous burst is bond relevant
                    burst_msgs = [m for _, m in current_burst]
                    if is_bond_trade_relevant(burst_msgs):
                        contexts.append({
                            "context_id": str(uuid.uuid4()),
                            "group_name": ch_name,
                            "messages": burst_msgs,
                            "message_indices": [i for i, _ in current_burst],
                            "snippet_text": format_context_snippet(burst_msgs, ch_name),
                        })
                    current_burst = []

            current_burst.append((idx, msg))

        if current_burst:
            burst_msgs = [m for _, m in current_burst]
            if is_bond_trade_relevant(burst_msgs):
                contexts.append({
                    "context_id": str(uuid.uuid4()),
                    "group_name": ch_name,
                    "messages": burst_msgs,
                    "message_indices": [i for i, _ in current_burst],
                    "snippet_text": format_context_snippet(burst_msgs, ch_name),
                })

    return contexts


def format_context_snippet(window: List[Dict[str, Any]], group_name: str) -> str:
    lines = []
    if group_name and group_name != "Default":
        lines.append(f"Channel: {group_name}")
    for m in window:
        time_str = m["dt"].strftime("%H:%M:%S")
        lines.append(f"[{time_str}] {m['sender']}: {m['message_text']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. LLM EXTRACTION & TOKEN / CONTEXT METRICS
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are an expert bond-trade extraction specialist for a fintech trading desk.
You are given a WhatsApp conversation window between counterparties / brokers negotiating a bond trade.

Your task is to extract structured bond trade details into a strict JSON object.

Fields required in JSON:
- bond_identity: string (e.g., "6.65% GS 2026", "9.60% Oxyzo June 27", "7.18% SDL 2028")
- isin: string or null (12-character ISIN if explicitly present in the text, e.g. "INE04VS07438", otherwise null)
- identity_type: one of "ISIN", "CUSIP", "SERIES", "NAME_ONLY", "UNKNOWN" (use "ISIN" if ISIN provided, "SERIES" or "NAME_ONLY" for named bonds)
- issuer_name: string or null (e.g. "Government of India", "Oxyzo", "State Government", "HUDCO")
- credit_rating: string or null (e.g. "AAA", "A+", "Sovereign", or null)
- maturity_date: ISO date "YYYY-MM-DD" or null (if year only like 2026, use "2026-12-31")
- coupon_rate: float or null (annual coupon rate e.g. 6.65, 9.60)
- face_value: float or null (face value per unit if mentioned, or null)
- currency: string ("INR" by default)
- buyer: string (name of the party or counterparty buying)
- seller: string (name of the party or counterparty selling)
- quantity: float or null (numeric quantity or deal size, e.g. 10.0, 5.0, 1.58, 50.0)
- quantity_unit: string (e.g. "Cr", "Crore", "Lakh", "Lacs", "Units")
- is_indicative: integer 0 (definite agreed trade) or 1 (indicative quote / inquiry)
- ytm: float or null (yield to maturity if negotiated, e.g. 6.465, 10.08)
- final_price: float or null (final agreed price or yield rate)
- settlement_reference: one of "T0", "T1", "T2" (default "T1" if unspecified or Monday value date)
- settlement_date: ISO date "YYYY-MM-DD" or null
- status: one of "CONCLUDED", "INCOMPLETE", "DISCARDED"
  * "CONCLUDED": Both parties explicitly agreed (e.g. "done", "closed", "ok lets close", "confirmed")
  * "INCOMPLETE": Trade negotiated or quoted but without final mutual agreement or confirmation
  * "DISCARDED": Rejected, out of market, or small talk
- is_deal_confirmed: boolean (true ONLY if both parties confirmed the trade)

Strict rules:
1. Ground every extracted field strictly in the conversation text. Do NOT hallucinate.
2. Return ONLY the JSON object, with no preamble or commentary.
"""


def get_model_context_length(model: str = MODEL_NAME) -> int:
    """Query model context window length from Ollama client."""
    try:
        client = ollama.Client()
        show_res = client.show(model)
        # Search modelinfo for context_length
        if hasattr(show_res, "modelinfo") and show_res.modelinfo:
            for k, v in show_res.modelinfo.items():
                if "context_length" in k.lower() and isinstance(v, int):
                    return v
        # Also check parameters if string like num_ctx
        if hasattr(show_res, "parameters") and show_res.parameters:
            ctx_match = re.search(r"num_ctx\s+(\d+)", show_res.parameters)
            if ctx_match:
                return int(ctx_match.group(1))
    except Exception as e:
        print(f"Notice: Could not fetch context length via client.show({model}): {e}")

    # Default fallback for gemma4:31b-cloud
    return 262144


def call_llm_extraction(context_snippet: str, model: str = MODEL_NAME) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calls langchain-ollama ChatOllama and extracts structured trade details.
    Returns:
    (extracted_json, token_usage_dict)
    """
    llm = ChatOllama(model=model, format="json", temperature=0.0)

    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nConversation Window:\n{context_snippet}"

    response = llm.invoke(prompt)

    # Extract real token counts from usage_metadata and response_metadata
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        prompt_tokens = response.usage_metadata.get("input_tokens", 0)
        completion_tokens = response.usage_metadata.get("output_tokens", 0)
        total_tokens = response.usage_metadata.get("total_tokens", prompt_tokens + completion_tokens)
    elif hasattr(response, "response_metadata") and response.response_metadata:
        meta = response.response_metadata
        prompt_tokens = meta.get("prompt_eval_count", 0)
        completion_tokens = meta.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

    token_metrics = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    # Clean markdown fences if any
    raw_content = response.content
    cleaned_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.MULTILINE).strip()

    try:
        extracted_data = json.loads(cleaned_content)
    except Exception:
        # Fallback regex search for JSON block
        json_match = re.search(r"(\{.*\})", cleaned_content, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group(1))
            except Exception:
                extracted_data = {"status": "DISCARDED", "is_deal_confirmed": False, "raw_error": raw_content}
        else:
            extracted_data = {"status": "DISCARDED", "is_deal_confirmed": False, "raw_error": raw_content}

    return extracted_data, token_metrics


# --------------------------------------------------------------------------
# 5. BUSINESS RULE VALIDATION
# --------------------------------------------------------------------------

def validate_extraction(deal: Any) -> Dict[str, Any]:
    """
    Validates business rules for an extracted trade:
    - ISIN format (12 alphanum chars)
    - Identity type check
    - Buyer != Seller
    - Positive price & quantity
    - Settlement reference check
    - Status consistency
    """
    if not isinstance(deal, dict):
        return {
            "status": "DISCARDED",
            "is_valid": False,
            "validation_errors": ["Invalid extraction output format"],
            "validation_warnings": [],
        }

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Status & Confirmation
    is_confirmed = deal.get("is_deal_confirmed", False)
    status = deal.get("status", "INCOMPLETE")
    if not is_confirmed and status == "CONCLUDED":
        deal["status"] = "INCOMPLETE"
        warnings.append("Mutual confirmation not found in text; status marked INCOMPLETE")

    # 2. Bond Identity
    bond_identity = deal.get("bond_identity")
    if not bond_identity or bond_identity == "UNKNOWN":
        errors.append("Bond identity missing or unknown")

    # 3. ISIN Format
    isin = deal.get("isin")
    if isin:
        isin = isin.strip().upper()
        deal["isin"] = isin
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", isin):
            warnings.append(f"ISIN '{isin}' does not strictly match 12-char international format")

    # 4. Identity Type CHECK constraint
    id_type = deal.get("identity_type", "UNKNOWN")
    if id_type not in ('ISIN', 'CUSIP', 'SERIES', 'NAME_ONLY', 'UNKNOWN'):
        deal["identity_type"] = "ISIN" if isin else "NAME_ONLY"

    # 5. Buyer & Seller
    buyer = deal.get("buyer")
    seller = deal.get("seller")

    if deal.get("status") == "CONCLUDED":
        if not buyer or buyer in ("UNKNOWN", "None", ""):
            errors.append("Missing buyer in concluded trade")
        if not seller or seller in ("UNKNOWN", "None", ""):
            errors.append("Missing seller in concluded trade")
        if buyer and seller and buyer.lower() == seller.lower():
            errors.append("Buyer and seller cannot be the same entity")

    # 6. Price & Quantity
    price = deal.get("final_price") or deal.get("price") or deal.get("ytm")
    if price is not None:
        try:
            deal["final_price"] = float(price)
            if deal["final_price"] <= 0:
                errors.append("Price/Yield must be positive")
        except (ValueError, TypeError):
            errors.append("Price must be a valid number")
    elif deal.get("status") == "CONCLUDED":
        errors.append("Missing agreed price/yield in concluded trade")

    qty = deal.get("quantity")
    if qty is not None:
        try:
            deal["quantity"] = float(qty)
            if deal["quantity"] <= 0:
                errors.append("Quantity must be positive")
        except (ValueError, TypeError):
            errors.append("Quantity must be a valid number")

    # 7. Settlement Reference CHECK constraint
    settlement_ref = deal.get("settlement_reference") or "T1"
    if settlement_ref not in ('T0', 'T1', 'T2'):
        deal["settlement_reference"] = "T1"

    # 8. Status CHECK constraint
    final_status = deal.get("status", "INCOMPLETE")
    if final_status not in ('INCOMPLETE', 'CONCLUDED', 'DISCARDED'):
        deal["status"] = "INCOMPLETE"

    is_valid = len(errors) == 0
    deal["is_valid"] = is_valid
    deal["validation_errors"] = errors
    deal["validation_warnings"] = warnings

    return deal
