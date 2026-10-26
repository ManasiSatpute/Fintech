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
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import ollama
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config
import privacy
from normalize import (
    is_valid_isin,
    normalize_maturity,
    normalize_quantity,
    normalize_yield_coupon,
    resolve_settlement,
)

MODEL_NAME = config.MODEL_NAME
ISSUER_ALIASES_PATH = Path(__file__).with_name("issuer_aliases.json")
with ISSUER_ALIASES_PATH.open(encoding="utf-8") as aliases_file:
    ISSUER_ALIASES = json.load(aliases_file)
if not isinstance(ISSUER_ALIASES, dict) or any(
    not isinstance(issuer, str)
    or not issuer.strip()
    or not isinstance(aliases, list)
    or not all(isinstance(alias, str) and alias.strip() for alias in aliases)
    for issuer, aliases in ISSUER_ALIASES.items()
):
    raise ValueError(f"Invalid issuer alias configuration: {ISSUER_ALIASES_PATH}")

LEXICON_PATH = Path(__file__).with_name("lexicon.json")


def load_lexicon(path: Path = LEXICON_PATH) -> Dict[str, List[str]]:
    # A fluent Marathi/Hindi speaker should review the Roman-script entries.
    with path.open(encoding="utf-8") as lexicon_file:
        lexicon = json.load(lexicon_file)
    if not isinstance(lexicon, dict) or any(
        not isinstance(category, str)
        or not isinstance(entries, list)
        or not all(isinstance(entry, str) and entry.strip() for entry in entries)
        for category, entries in lexicon.items()
    ):
        raise ValueError(f"Invalid lexicon configuration: {path}")
    if "needs_context" not in lexicon:
        raise ValueError(f"Lexicon must define needs_context: {path}")
    return lexicon


LEXICON = load_lexicon()
EXTRACTION_EXAMPLES_PATH = Path(__file__).with_name("extraction_examples.json")
with EXTRACTION_EXAMPLES_PATH.open(encoding="utf-8") as examples_file:
    EXTRACTION_EXAMPLES = json.load(examples_file)
if not isinstance(EXTRACTION_EXAMPLES, list) or len(EXTRACTION_EXAMPLES) != 10:
    raise ValueError(
        f"Expected ten extraction examples in {EXTRACTION_EXAMPLES_PATH}"
    )


def _normalize_lexicon_text(value: Any) -> str:
    text = _clean_message_text(value).translate(
        str.maketrans("०१२३४५६७८९", "0123456789")
    ).casefold()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def match_lexicon(message: Any) -> Dict[str, Any]:
    """Return lexical intent hints only; matches are not extraction decisions."""
    raw_text = (
        message.get("message_text", "")
        if isinstance(message, dict)
        else message
    )
    normalized_text = _normalize_lexicon_text(raw_text)
    ambiguous_entries = {
        _normalize_lexicon_text(entry)
        for entry in LEXICON.get("needs_context", [])
    }
    matches: Dict[str, List[str]] = {}
    for category, entries in LEXICON.items():
        found = []
        for entry in entries:
            normalized_entry = _normalize_lexicon_text(entry)
            if category != "needs_context" and normalized_entry in ambiguous_entries:
                continue
            if re.search(
                rf"(?<!\w){re.escape(normalized_entry)}(?!\w)",
                normalized_text,
                flags=re.UNICODE,
            ):
                found.append(entry)
        if found:
            matches[category] = found
    return {
        "matches": matches,
        "needs_context": matches.get("needs_context", []),
        "has_intent_hints": any(
            category != "needs_context" for category in matches
        ),
    }


def format_lexicon_glossary() -> str:
    """Build a compact glossary for the extraction prompt."""
    sections = []
    for category, entries in LEXICON.items():
        if category == "needs_context":
            continue
        sections.append(f"{category}: {', '.join(entries)}")
    sections.append(
        "needs_context (ambiguous; interpret only with surrounding messages): "
        + ", ".join(LEXICON["needs_context"])
    )
    return " | ".join(sections)

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

_DAY_FIRST_TIMESTAMP_FORMATS = [
    "%d/%m/%y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%y %I:%M %p",
    "%d/%m/%Y %I:%M %p",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%d/%m/%Y %H:%M",
]
_MONTH_FIRST_TIMESTAMP_FORMATS = [
    "%m/%d/%y %I:%M:%S %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
]
TIMESTAMP_FORMATS = (
    _DAY_FIRST_TIMESTAMP_FORMATS + _MONTH_FIRST_TIMESTAMP_FORMATS
    if config.DAY_FIRST_DEFAULT
    else _MONTH_FIRST_TIMESTAMP_FORMATS + _DAY_FIRST_TIMESTAMP_FORMATS
)


def _timestamp_formats(day_first: bool) -> List[str]:
    day_formats = [
        "%d/%m/%y %I:%M:%S %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%y %I:%M %p",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %H:%M",
    ]
    month_formats = [
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %H:%M",
    ]
    return day_formats + month_formats if day_first else month_formats + day_formats


def _infer_day_first(timestamps: List[Any]) -> bool:
    day_first_evidence = 0
    month_first_evidence = 0
    date_pattern = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/")
    for timestamp in timestamps:
        if not isinstance(timestamp, str):
            continue
        match = date_pattern.match(timestamp)
        if not match:
            continue
        first, second = (int(value) for value in match.groups())
        if first > 12:
            day_first_evidence += 1
        elif second > 12:
            month_first_evidence += 1
    if day_first_evidence == month_first_evidence:
        return config.DAY_FIRST_DEFAULT
    return day_first_evidence > month_first_evidence


def _parse_timestamp(raw_timestamp: Any, day_first: bool) -> Optional[datetime]:
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        return None
    timestamp = raw_timestamp.strip().replace(",", "")
    for timestamp_format in _timestamp_formats(day_first):
        try:
            return datetime.strptime(timestamp, timestamp_format)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw_timestamp.strip())
    except ValueError:
        return None


def parse_timestamp_str(date_str: str, time_str: str) -> Optional[datetime]:
    combined = f"{date_str.strip()} {time_str.strip()}"
    combined = re.sub(r"\s+", " ", combined)
    return _parse_timestamp(combined, config.DAY_FIRST_DEFAULT)


def _clean_message_text(value: Any) -> str:
    if not isinstance(value, str) or value.strip().casefold() == "none":
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )


def _normalize_display_name(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_message_text(value)).strip()


def _device_free_number(value: Any) -> str:
    number = _normalize_display_name(value)
    return re.sub(r":\d+$", "", number)


def _chat_type(value: Any) -> str:
    return _normalize_display_name(value).upper()


def _is_own_sender(sender: str, direction: Any) -> bool:
    own_names = {name.casefold() for name in config.OWN_SENDER_NAMES}
    if sender.casefold() in own_names:
        return True
    if own_names:
        return False
    return _normalize_display_name(direction).upper() == "OUTGOING"


def _private_counterpart(item: Dict[str, Any]) -> str:
    sender_number = _device_free_number(item.get("sender_number"))
    receiver_number = _device_free_number(item.get("receiver_number"))
    sender_name = _normalize_display_name(
        item.get("sender_name") or item.get("sender")
    )
    own_names = {name.casefold() for name in config.OWN_SENDER_NAMES}
    if own_names:
        sender_is_own = sender_name.casefold() in own_names
    else:
        sender_is_own = _normalize_display_name(item.get("direction")).upper() == "OUTGOING"
    counterpart = receiver_number if sender_is_own else sender_number
    if not counterpart:
        counterpart = sender_number or receiver_number
    return counterpart


def _stable_message_chat_identity(record: Dict[str, Any]) -> str:
    chat_type = _chat_type(record.get("chat_type"))
    sender_number = _device_free_number(record.get("sender_number"))
    receiver_number = _device_free_number(record.get("receiver_number"))

    if chat_type == "PRIVATE":
        identity = [
            chat_type,
            sorted(number for number in (sender_number, receiver_number) if number),
        ]
    elif chat_type == "GROUP" and "UNKNOWN" not in record["chat_key"]:
        return record["chat_key"]
    else:
        identity = [
            chat_type,
            _normalize_display_name(record.get("chat_name")),
            sender_number,
            receiver_number,
        ]
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _is_media_placeholder(text: str) -> Tuple[bool, Optional[str]]:
    match = re.fullmatch(r"\[\s*(media|image|chat)\s+message\s*\]", text.strip(), re.I)
    if not match:
        return False, None
    return True, match.group(1).lower()


def parse_raw_chat_content(
    raw_content: str,
    chat_id: Optional[str] = None,
    return_report: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
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

    Existing callers receive the message list. Set ``return_report`` to receive
    ``(messages, report)`` without changing the message-list contract.
    """
    if not chat_id:
        chat_id = str(uuid.uuid4())

    content = raw_content.replace("\x00", "")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines:
        empty_report = {
            "parsed": 0,
            "skipped": 0,
            "media": 0,
            "per_chat_type": {},
            "distinct_chat_keys": 0,
        }
        return ([], empty_report) if return_report else []

    parsed_rows: List[Tuple[int, Dict[str, Any]]] = []
    skipped_count = 0
    jsonl_detected = False
    for line in lines:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and any(
            key in candidate
            for key in ("message", "message_text", "sender_name", "timestamp", "ts")
        ):
            jsonl_detected = True
            break

    if jsonl_detected:
        raw_rows = []
        timestamps = []
        for idx, line in enumerate(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                continue
            if not isinstance(item, dict):
                skipped_count += 1
                continue
            raw_rows.append((idx, item))
            timestamps.append(item.get("timestamp") or item.get("ts"))

        day_first = _infer_day_first(timestamps)
        incoming_names: Dict[str, Counter] = {}
        for _, item in raw_rows:
            if _chat_type(item.get("chat_type")) != "PRIVATE":
                continue
            counterpart = _private_counterpart(item)
            if not counterpart:
                continue
            key = f"PRIVATE:{counterpart}"
            sender = _normalize_display_name(
                item.get("sender_name") or item.get("sender")
            )
            if not _is_own_sender(sender, item.get("direction")):
                sender_name = _normalize_display_name(item.get("sender_name"))
                if sender_name:
                    incoming_names.setdefault(key, Counter())[sender_name] += 1

        for idx, item in raw_rows:
            raw_timestamp = item.get("timestamp") or item.get("ts")
            dt = _parse_timestamp(raw_timestamp, day_first)
            if dt is None:
                skipped_count += 1
                continue

            chat_type = _normalize_display_name(item.get("chat_type"))
            normalized_type = _chat_type(chat_type)
            sender_number = _normalize_display_name(item.get("sender_number"))
            receiver_number = _normalize_display_name(item.get("receiver_number"))
            sender = _normalize_display_name(
                item.get("sender_name") or item.get("sender") or sender_number
            )
            raw_group_name = item.get("group_name")
            normalized_group_name = _normalize_display_name(raw_group_name)
            group_name = raw_group_name or chat_type or "Direct"
            if normalized_type == "GROUP":
                group_key = _device_free_number(sender_number)
                if not group_key.isdigit():
                    group_key = ""
                chat_key = (
                    f"GROUP:{group_key}"
                    if group_key
                    else f"GROUP:UNKNOWN:{chat_id}"
                )
                chat_name = normalized_group_name or _normalize_display_name(chat_type)
            elif normalized_type == "PRIVATE":
                counterpart = _private_counterpart(item)
                chat_key = (
                    f"PRIVATE:{counterpart}"
                    if counterpart
                    else f"PRIVATE:UNKNOWN:{idx}"
                )
                names = incoming_names.get(chat_key, Counter())
                chat_name = (
                    min(names, key=lambda name: (-names[name], name.casefold()))
                    if names
                    else counterpart or chat_type or "Private chat"
                )
            else:
                chat_key = f"{normalized_type or 'UNKNOWN'}:{chat_id}"
                chat_name = raw_group_name or chat_type or "Direct"

            message_text = _clean_message_text(
                item.get("message") or item.get("message_text")
            )
            is_media, media_kind = _is_media_placeholder(message_text)
            if is_media:
                message_text = ""

            replying_to = _clean_message_text(item.get("replying_to"))
            record = {
                "chat_id": chat_id,
                "sender": sender,
                "message_text": message_text,
                "timestamp": dt.isoformat(),
                "dt": dt,
                "group_name": group_name,
                "channel": group_name,
                "direction": item.get("direction"),
                "chat_type": item.get("chat_type"),
                "sender_number": item.get("sender_number"),
                "receiver_number": item.get("receiver_number"),
                "replying_to": replying_to,
                "chat_key": chat_key,
                "chat_name": chat_name,
                "is_own": _is_own_sender(sender, item.get("direction")),
                "is_media": is_media,
                "media_kind": media_kind,
                "_source_index": idx,
            }
            parsed_rows.append((idx, record))
    else:
        day_first = _infer_day_first([
            f"{match.group('date')} {match.group('time')}"
            for line in lines
            for match in [LINE_RE_IOS.match(line) or LINE_RE_ANDROID.match(line)]
            if match
        ])
        for idx, line in enumerate(lines):
            m_ios = LINE_RE_IOS.match(line)
            m_android = LINE_RE_ANDROID.match(line)
            m = m_ios or m_android

            if m:
                dt = _parse_timestamp(
                    f"{m.group('date')} {m.group('time')}",
                    day_first,
                )
                if dt is None:
                    skipped_count += 1
                    continue
                sender = m.group("sender").strip()
                text = _clean_message_text(m.group("msg"))
                is_media, media_kind = _is_media_placeholder(text)
                if is_media:
                    text = ""
                record = {
                    "chat_id": chat_id,
                    "sender": sender,
                    "message_text": text,
                    "timestamp": dt.isoformat(),
                    "dt": dt,
                    "group_name": "Default",
                    "channel": "Default",
                    "direction": "",
                    "chat_type": "",
                    "sender_number": "",
                    "receiver_number": "",
                    "replying_to": "",
                    "chat_key": f"CHAT:{chat_id}",
                    "chat_name": "Default",
                    "is_own": _is_own_sender(sender, ""),
                    "is_media": is_media,
                    "media_kind": media_kind,
                    "_source_index": idx,
                }
                parsed_rows.append((idx, record))
            elif parsed_rows:
                parsed_rows[-1][1]["message_text"] += "\n" + _clean_message_text(line)

    parsed_rows.sort(key=lambda row: (row[1]["dt"], row[0]))
    occurrences: Counter = Counter()
    messages = []
    chat_type_counts: Counter = Counter()
    media_count = 0
    for _, record in parsed_rows:
        record["is_system"] = is_system_message(record)
        record["noise_label"] = get_noise_label(record)
        chat_identity = _stable_message_chat_identity(record)
        identity = json.dumps(
            [
                chat_identity,
                record["timestamp"],
                _normalize_display_name(record["direction"]),
                record["sender"],
                _device_free_number(record["sender_number"]),
                _device_free_number(record["receiver_number"]),
                record["replying_to"],
                record["message_text"],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        record["message_id"] = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:{occurrence}")
        )
        record.pop("_source_index")
        chat_type_counts[record["chat_type"] or "UNKNOWN"] += 1
        media_count += int(record["is_media"])
        messages.append(record)

    parse_report = {
        "parsed": len(messages),
        "skipped": skipped_count,
        "media": media_count,
        "per_chat_type": dict(sorted(chat_type_counts.items())),
        "distinct_chat_keys": len({message["chat_key"] for message in messages}),
    }
    return (messages, parse_report) if return_report else messages


# --------------------------------------------------------------------------
# 2. NOISE FILTER & ROUTING
# --------------------------------------------------------------------------

SYSTEM_SENDERS = {"Chat", "System"}
_SYSTEM_NAME = r"[^\W\d_][\w .'-]*"
SYSTEM_EVENT_PATTERNS = [
    re.compile(
        r"^messages and calls are end-to-end encrypted"
        r"(?:\.? only people in this chat can read, listen to, or share them\.?)?$",
        re.I,
    ),
    re.compile(r"^(?:you|.+?) created group .+$", re.I),
    re.compile(r"^.+ changed the subject(?: to .+)?$", re.I),
    re.compile(r"^.+ changed (?:this|the) group['’]s icon$", re.I),
    re.compile(r"^(?:your )?security code(?: with .+)? changed\.?$", re.I),
    re.compile(rf"^{_SYSTEM_NAME} added {_SYSTEM_NAME}$", re.I),
    re.compile(rf"^{_SYSTEM_NAME} left$", re.I),
    re.compile(rf"^{_SYSTEM_NAME} joined(?: .+)?$", re.I),
]

ACK_RE = re.compile(r"^(?:ok|okay|yes|sure|done|closed|s|np|good|haan)[.!?]*$", re.I)
GREETING_PATTERNS = [
    r"^(?:hello|hi|hey)\b(?:\s+[A-Za-z][A-Za-z ,'-]*)?[!.?]*$",
    r"^good\s+(?:morning|afternoon|evening)(?:\s+[A-Za-z][A-Za-z ,'-]*)?[!.?]*$",
]
CHITCHAT_PATTERNS = [
    r"^how are you(?: doing)?[!.?]*$",
    r"^how(?:'s| is) it going[!.?]*$",
    r"^what is going on[!.?]*$",
    r"^see you(?: later)?[!.?]*$",
    r"^nice[!.?]*$",
    r"^all good[!.?]*$",
    r"^that['’]s great[!.?]*$",
    r"^had lunch\??$",
    r"^thanks[!.?]*$",
    r"^thank you[!.?]*$",
]
GREETING_RE = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)
CHITCHAT_RE = re.compile("|".join(CHITCHAT_PATTERNS), re.IGNORECASE)

BOND_KEYWORDS_RE = re.compile(
    r"\b(NCD|SDL|G[- ]?Sec|GS|T[- ]?bill|tax[- ]?free|debenture|coupon|maturity)\b",
    re.IGNORECASE
)

ISIN_CANDIDATE_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9,11}\b", re.IGNORECASE)
PARTIAL_ISIN_RE = re.compile(
    r"\bISIN\b\s*[:#-]?\s*(?:[A-Z]{2}\s*)?(\d{1,11})\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(
    r"\b(?:"
    r"\d[\d,]*(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d[\d,]*(?:\.\d+)?)?"
    r"\s*(?:crs?|crores?|lacs?|lakhs?|l|units)\b"
    r"|multiple\s+of\b|entire\s+quantum\b"
    r")",
    re.IGNORECASE,
)
YIELD_PATTERNS = [
    re.compile(r"@\s*\d{1,2}(?:\.\d+)?\s*%?", re.IGNORECASE),
    re.compile(r"\b\d{1,2}(?:\.\d+)?\s*%", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\.\d{1,4}\s*[-/]\s*\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*bps?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:lvl|level|ytm)\b(?:\s*[:=@]?\s*\d{1,2}(?:\.\d+)?\s*%?)?",
        re.IGNORECASE,
    ),
]
ACTION_RE = re.compile(
    r"\b(?:bids?|offers?|need|required|take|close|closed|done|pass|firm|sold|"
    r"bought|mine|yours|holding|lena\s+hai)\b",
    re.IGNORECASE,
)
SETTLEMENT_RE = re.compile(
    r"\b(?:T\s*\+\s*[01]|T[01]|today|tom|tomorrow|kal|aaj|VD|value\s+date|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)

_RELEVANCE_WEIGHTS = {
    "valid_isin": 1,
    "partial_isin": 1,
    "amount": 2,
    "yield": 2,
    "action": 1,
    "settlement": 1,
    "bond_vocabulary": 1,
    "issuer_alias": 2,
}


def _isin_checksum_valid(candidate: str) -> bool:
    expanded = "".join(
        str(ord(character.upper()) - ord("A") + 10)
        if character.isalpha()
        else character
        for character in candidate
    )
    digits = [int(digit) for digit in expanded]
    checksum = 0
    double_digit = False
    for digit in reversed(digits):
        value = digit * 2 if double_digit else digit
        checksum += value // 10 + value % 10
        double_digit = not double_digit
    return checksum % 10 == 0


def _unique_matches(pattern: re.Pattern, text: str) -> List[str]:
    matches = []
    seen = set()
    for match in pattern.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        normalized = value.casefold()
        if normalized not in seen:
            seen.add(normalized)
            matches.append(value)
    return matches


def compute_hints(message: Any) -> Dict[str, Any]:
    """Extract deterministic bond-trade signals from a message or message body."""
    raw_text = message.get("message_text", "") if isinstance(message, dict) else message
    text = _clean_message_text(raw_text)
    candidates = _unique_matches(ISIN_CANDIDATE_RE, text)
    valid_isins = [
        candidate
        for candidate in candidates
        if len(candidate) == 12 and candidate[-1].isdigit()
        and _isin_checksum_valid(candidate)
    ]
    invalid_isins = [candidate for candidate in candidates if candidate not in valid_isins]
    partial_match = PARTIAL_ISIN_RE.search(text)
    amount_matches = _unique_matches(AMOUNT_RE, text)
    yield_matches = []
    seen_yields = set()
    for pattern in YIELD_PATTERNS:
        for value in _unique_matches(pattern, text):
            if value.casefold() not in seen_yields:
                seen_yields.add(value.casefold())
                yield_matches.append(value)
    action_matches = _unique_matches(ACTION_RE, text)
    settlement_matches = _unique_matches(SETTLEMENT_RE, text)
    vocabulary_matches = _unique_matches(BOND_KEYWORDS_RE, text)
    alias_matches = []
    normalized_text = text.casefold()
    for issuer, aliases in ISSUER_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_text, re.I):
                alias_matches.append(issuer)
                break

    signals = {
        "valid_isin": bool(valid_isins),
        "partial_isin": partial_match is not None,
        "amount": bool(amount_matches),
        "yield": bool(yield_matches),
        "action": bool(action_matches),
        "settlement": bool(settlement_matches),
        "bond_vocabulary": bool(vocabulary_matches),
        "issuer_alias": bool(alias_matches),
    }
    relevance_score = sum(
        weight for signal, weight in _RELEVANCE_WEIGHTS.items() if signals[signal]
    )
    return {
        "valid_isin": signals["valid_isin"],
        "valid_isins": valid_isins,
        "invalid_isin_candidates": invalid_isins,
        "partial_isin": partial_match.group(1) if partial_match else None,
        "amount": amount_matches,
        "yield": yield_matches,
        "action": action_matches,
        "settlement": settlement_matches,
        "bond_vocabulary": vocabulary_matches,
        "issuer_alias": alias_matches,
        "signals": signals,
        "relevance_score": relevance_score,
    }


def is_system_message(msg: Dict[str, Any]) -> bool:
    sender = msg.get("sender", "")
    text = msg.get("message_text", "").strip()
    if sender in SYSTEM_SENDERS:
        return True
    return any(pattern.fullmatch(text) for pattern in SYSTEM_EVENT_PATTERNS)


def is_system_or_media(msg: Dict[str, Any]) -> bool:
    return is_system_message(msg) or bool(msg.get("is_media", False))


def is_chit_chat(msg: Dict[str, Any]) -> bool:
    text = msg.get("message_text", "").strip()
    return bool(GREETING_RE.fullmatch(text) or CHITCHAT_RE.fullmatch(text))


def get_noise_label(msg: Dict[str, Any]) -> str:
    if is_system_message(msg):
        return "admin"
    text = msg.get("message_text", "").strip()
    if ACK_RE.fullmatch(text):
        return "ack"
    if GREETING_RE.fullmatch(text):
        return "greeting"
    if is_chit_chat(msg):
        return "chitchat"
    return "none"


# --------------------------------------------------------------------------
# 3. GROUPING INTO TRADE CONTEXTS
# --------------------------------------------------------------------------

WINDOW_GAP = timedelta(minutes=config.GAP_MINUTES)


def is_bond_trade_relevant(window: List[Dict[str, Any]]) -> bool:
    return get_window_relevance_score(window) >= config.RELEVANCE_THRESHOLD


def get_window_relevance_score(window: List[Dict[str, Any]]) -> int:
    text = "\n".join(
        message.get("message_text", "")
        for message in window
        if not message.get("is_media", False)
    )
    return compute_hints(text)["relevance_score"]


def _normalize_reply_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", _clean_message_text(value)).casefold(),
    ).strip()


_REPLY_MEDIA_PLACEHOLDER_RE = re.compile(
    r"^\[\s*(?:media|image|chat)\s+message\s*\]$", re.IGNORECASE
)


def _reply_target_index(
    replying_to: Any,
    earlier_messages: List[Tuple[int, Dict[str, Any]]],
) -> Optional[int]:
    quote = _normalize_reply_text(replying_to)
    if not quote:
        return None
    if _REPLY_MEDIA_PLACEHOLDER_RE.fullmatch(quote):
        for index, message in reversed(earlier_messages):
            if message.get("is_media", False):
                return index
        return None

    for index, message in reversed(earlier_messages):
        earlier_text = _normalize_reply_text(message.get("message_text"))
        if earlier_text and (
            quote == earlier_text or quote in earlier_text or earlier_text in quote
        ):
            return index
    return None


def assign_contexts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign messages to chronological windows, honoring reply links."""
    chats: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, message in enumerate(messages):
        chat_key = message.get("chat_key") or (
            f"CHAT:{message.get('chat_id', 'default')}"
        )
        chats.setdefault(chat_key, []).append((index, message))

    windows: List[Dict[str, Any]] = []
    for chat_key, chat_messages in chats.items():
        chat_messages.sort(key=lambda item: (item[1]["dt"], item[0]))
        assigned_by_message: Dict[int, Dict[str, Any]] = {}
        earlier_messages: List[Tuple[int, Dict[str, Any]]] = []
        last_message_dt = None
        active_window: Optional[Dict[str, Any]] = None
        most_recent_window: Optional[Dict[str, Any]] = None
        chat_windows: List[Dict[str, Any]] = []

        for index, message in chat_messages:
            if message.get("is_system", False) or is_system_message(message):
                continue

            if (
                last_message_dt is not None
                and message["dt"] - last_message_dt > WINDOW_GAP
            ):
                active_window = None

            reply_target = _reply_target_index(
                message.get("replying_to"),
                earlier_messages,
            )
            if reply_target is not None:
                target_window = assigned_by_message.get(reply_target)
                if target_window is not None:
                    target_window["messages"].append(message)
                    target_window["message_indices"].append(index)
                    target_window["reply_linked_count"] += 1
                    target_window["end"] = max(target_window["end"], message["dt"])
                    assigned_by_message[index] = target_window
                    earlier_messages.append((index, message))
                    last_message_dt = message["dt"]
                    continue

            reply_text = _clean_message_text(message.get("replying_to"))
            unresolved_reply = None
            if reply_text:
                unresolved_reply = {
                    "message_id": message.get("message_id"),
                    "replying_to": reply_text,
                }

            if (
                not reply_text
                and message.get("noise_label") == "ack"
                and active_window is None
            ):
                if most_recent_window is not None:
                    most_recent_window["messages"].append(message)
                    most_recent_window["message_indices"].append(index)
                    most_recent_window["end"] = max(
                        most_recent_window["end"],
                        message["dt"],
                    )
                    assigned_by_message[index] = most_recent_window
                earlier_messages.append((index, message))
                last_message_dt = message["dt"]
                continue

            if active_window is None:
                chat_name = (
                    message.get("chat_name")
                    or message.get("group_name")
                    or str(chat_key)
                )
                active_window = {
                    "chat_key": chat_key,
                    "chat_name": chat_name,
                    "chat_type": message.get("chat_type", ""),
                    "messages": [],
                    "message_indices": [],
                    "reply_linked_count": 0,
                    "unresolved_replies": [],
                }
                windows.append(active_window)
                chat_windows.append(active_window)
            active_window["messages"].append(message)
            active_window["message_indices"].append(index)
            active_window["end"] = message["dt"]
            if "start" not in active_window:
                active_window["start"] = message["dt"]
            if unresolved_reply is not None:
                active_window["unresolved_replies"].append(unresolved_reply)
            assigned_by_message[index] = active_window
            earlier_messages.append((index, message))
            last_message_dt = message["dt"]
            most_recent_window = active_window

        index_by_identity = {
            id(message): index for index, message in chat_messages
        }
        for window in chat_windows:
            window["messages"].sort(
                key=lambda message: (
                    message["dt"],
                    index_by_identity[id(message)],
                )
            )
            window["message_indices"] = [
                index_by_identity[id(message)] for message in window["messages"]
            ]
            window["start"] = min(message["dt"] for message in window["messages"])
            window["end"] = max(message["dt"] for message in window["messages"])
            window["message_ids"] = [
                message.get("message_id") for message in window["messages"]
            ]

    return sorted(windows, key=lambda window: window["start"])


def group_into_trade_contexts(
    messages: List[Dict[str, Any]],
    return_skipped: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Assigns messages with ``assign_contexts`` and separates low-relevance windows.
    """
    contexts = []
    skipped_low_relevance = []
    for assigned in assign_contexts(messages):
        burst_messages = assigned["messages"]
        group_name = assigned["chat_name"]
        relevance_score = get_window_relevance_score(burst_messages)
        privacy_map: Dict[str, str] = {}
        snippet_text, message_number_map = format_context_snippet(
            burst_messages,
            group_name,
            privacy_map,
        )
        context = {
            "context_id": str(uuid.uuid4()),
            "chat_key": assigned["chat_key"],
            "chat_name": assigned["chat_name"],
            "chat_type": assigned["chat_type"],
            "message_ids": assigned["message_ids"],
            "start": assigned["start"].isoformat(),
            "end": assigned["end"].isoformat(),
            "reply_linked_count": assigned["reply_linked_count"],
            "unresolved_replies": assigned["unresolved_replies"],
            "group_name": group_name,
            "messages": burst_messages,
            "message_indices": assigned["message_indices"],
            "snippet_text": snippet_text,
            "privacy_map": privacy_map,
            "message_number_map": message_number_map,
            "relevance_score": relevance_score,
        }
        if relevance_score >= config.RELEVANCE_THRESHOLD:
            contexts.append(context)
        else:
            skipped_low_relevance.append(context)

    if return_skipped:
        return contexts, skipped_low_relevance
    return contexts


def format_context_snippet(
    window: List[Dict[str, Any]],
    group_name: str,
    placeholder_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if not window:
        return "", {}

    first_message = window[0]
    chat_name = first_message.get("chat_name") or group_name or "Default"
    chat_type = first_message.get("chat_type") or "UNKNOWN"
    date_text = first_message["dt"].strftime("%Y-%m-%d")
    lines = [
        f"Chat: {chat_name}",
        f"Type: {chat_type}",
        f"Date: {date_text}",
    ]
    message_number_map = {}
    private_numbers = [
        str(message.get("sender_number", ""))
        for message in window
        if message.get("sender_number")
    ]
    private_numbers.extend(
        str(message.get("receiver_number", ""))
        for message in window
        if message.get("receiver_number")
    )

    for number, message in enumerate(window, start=1):
        message_number_map[f"#{number}"] = message.get("message_id")
        time_text = message["dt"].strftime("%H:%M:%S")
        sender = _clean_message_text(message.get("sender", ""))
        if message.get("is_own", False):
            sender = f"[OWN] {sender}"
        if message.get("is_media", False):
            body_lines = ["[attachment]"]
        else:
            body = _clean_message_text(message.get("message_text", ""))
            body_lines = body.splitlines() or [""]
        lines.append(f"#{number} [{time_text}] {sender}: {body_lines[0]}")
        lines.extend(f"    {body_line}" for body_line in body_lines[1:])

        reply = _clean_message_text(message.get("replying_to", ""))
        if reply:
            quoted = reply[:200]
            lines.append(f"    replying to: {quoted}")

    snippet = "\n".join(lines)
    if config.PII_MASKING_ENABLED:
        person_names = [
            message.get("sender", "")
            for message in window
            if message.get("sender")
        ]
        if chat_type == "PRIVATE" and chat_name:
            person_names.append(chat_name)
        masked_snippet, mapped_placeholders = privacy.mask_pii(
            snippet,
            sender_numbers=private_numbers,
            person_names=person_names,
            mask_person_names=config.MASK_PERSON_NAMES,
        )
        if placeholder_map is not None:
            placeholder_map.update(mapped_placeholders)
        snippet = masked_snippet
    return snippet, message_number_map


# --------------------------------------------------------------------------
# 4. LLM EXTRACTION & TOKEN / CONTEXT METRICS
# --------------------------------------------------------------------------

PROMPT_VERSION = "bond-extraction-v2"
EXTRACTION_STATUSES = {
    "INITIATED",
    "NEGOTIATING",
    "CONCLUDED",
    "CANCELLED",
    "UNCERTAIN",
    "DISCARDED",
}


class DealExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bond_identity_text: Optional[str] = None
    isin_text: Optional[str] = None
    partial_isin_text: Optional[str] = None
    issuer_text: Optional[str] = None
    maturity_text: Optional[str] = None
    coupon_rate: Optional[float] = None
    credit_rating: Optional[str] = None
    quantity_text: Optional[str] = None
    ytm: Optional[Union[float, str]] = None
    ytm_is_two_way: Optional[bool] = None
    price: Optional[float] = None
    settlement_text: Optional[str] = None
    owner_side: Optional[Literal["BUY", "SELL", "UNKNOWN"]] = None
    buyer_text: Optional[str] = None
    seller_text: Optional[str] = None
    rfq_id: Optional[str] = None
    package_note: Optional[str] = None
    is_indicative: Optional[bool] = None
    status: Optional[
        Literal[
            "INITIATED",
            "NEGOTIATING",
            "CONCLUDED",
            "CANCELLED",
            "UNCERTAIN",
            "DISCARDED",
        ]
    ] = None
    status_evidence: Optional[str] = None
    discard_reason: Optional[str] = None
    evidence: Optional[Dict[str, List[str]]] = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deals: List[DealExtraction] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You extract bond-trade events from a WhatsApp conversation window.
Return exactly one JSON object matching the supplied schema: {"deals": [...]}.
One broadcast mentioning several bonds must produce one deal per bond. A shared
yield or other shared term belongs in package_note, and must not be copied into
each bond's fields unless the text supports doing so.

Every schema field is nullable. Use null whenever the conversation does not
state a value. Preserve text exactly as written for *_text fields. Do not
convert maturity to a date, derive quantities, or infer any missing term.
bond_identity_text requires bond-specific identity evidence; an issuer name
alone is never a bond identity. Set price only for an explicitly stated clean
price, never for a yield. A number after an ISIN joined by a hyphen (for example "<ISIN> - 43") is a unit count, never a yield.

For every non-null value, evidence maps its field name to one or more exact
numbered conversation lines such as ["#2", "#4"]. Do not invent line numbers.

Trading semantics (direction words are always relative to who spoke):
- A request for an offer ("offer pls", "offer please", "need offer",
  "any offer?", "offers?") means the sender wants to BUY; a request for bids
  ("bid pls", "bid please", "any bids?") means the sender wants to SELL.
- A level-bearing offer ("offer at X", "I have an offer", or a quoted
  percentage followed by "offer") means the sender is SELLING. A level-bearing
  bid ("bid at X", "I have a bid") means the sender is BUYING.
- "holding bid at X" is a firm buy level; "can take" is a buy level.
- owner_side describes BUY/SELL relative to the [OWN] participant, not the
  speaker. Use UNKNOWN if direction cannot be established. buyer_text and
  seller_text must be copied exactly as written when stated.
- If a message only gives an ISIN, units, and settlement as an order
  instruction without stating a side, leave buyer_text and seller_text null
  and set owner_side to UNKNOWN.

Status rules:
- INITIATED: an offer or inquiry has no response.
- NEGOTIATING: counterparties exchanged counter-levels without acceptance.
- CONCLUDED requires identifiable key terms and later acceptance from the
  other side (for example, "ok lets close at 10.08" or "closed"). A bare "ok"
  unrelated to the trade is not acceptance.
- CANCELLED requires an explicit refusal from the other participant replying
  to a specific offer or inquiry, either through a reply link or as the next
  message in the same chat. Use the refusal line as status_evidence. Refusal
  words include the configured pass_or_refusal glossary entries. If the
  refusal cannot be tied to a specific offer or inquiry, use UNCERTAIN.
  Silence remains INITIATED.
- UNCERTAIN when evidence is ambiguous or insufficient to classify otherwise.
- DISCARDED for operations, payments, allocation, validation, "reported" or
  "accepted" processing notices, and banter. Give a concise discard_reason.
- When levels are revised, use the last stated level. Keep status_evidence
  grounded in the numbered lines.

Never infer a bond from an issuer name alone. Never invent values. Treat
reported/accepted operational events as DISCARDED, not a trade conclusion.
Return only the schema-conforming JSON, without prose or markdown fences.
"""


def _extraction_prompt(context_snippet: str, previous_error: Optional[str] = None) -> str:
    prompt = (
        f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
        "Compact conversation glossary (intent hints only, not decisions):\n"
        f"{format_lexicon_glossary()}\n\n"
        "Synthetic examples (illustrative only):\n"
        f"{json.dumps(EXTRACTION_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        f"Conversation Window:\n{context_snippet}"
    )
    if previous_error:
        prompt += (
            "\n\nYour previous response failed schema/JSON validation. "
            "Correct the response using this error; do not repeat the failure:\n"
            f"{previous_error}"
        )
    return prompt


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
    return config.MODEL_CONTEXT_LENGTH_FALLBACK


def call_llm_extraction(
    context_snippet: str,
    model: str = MODEL_NAME,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return structured deals, or an explicit extraction_failed result."""
    llm = ChatOllama(
        model=model,
        format=ExtractionResult.model_json_schema(),
        temperature=0.0,
    )
    prompt = _extraction_prompt(context_snippet)
    token_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "attempts": 0,
    }
    last_raw_text = ""
    last_error = ""

    for attempt in range(config.LLM_RETRIES + 1):
        token_metrics["attempts"] += 1
        try:
            response = llm.invoke(prompt)
            usage = getattr(response, "usage_metadata", None) or {}
            metadata = getattr(response, "response_metadata", None) or {}
            prompt_tokens = usage.get(
                "input_tokens",
                metadata.get("prompt_eval_count", 0),
            )
            completion_tokens = usage.get(
                "output_tokens",
                metadata.get("eval_count", 0),
            )
            token_metrics["prompt_tokens"] += prompt_tokens
            token_metrics["completion_tokens"] += completion_tokens
            token_metrics["total_tokens"] += usage.get(
                "total_tokens",
                prompt_tokens + completion_tokens,
            )
            content = getattr(response, "content", response)
            last_raw_text = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, default=str)
            )
            parsed = json.loads(last_raw_text)
            validated = ExtractionResult.model_validate(parsed)
            result = validated.model_dump(mode="json")
            result["prompt_version"] = PROMPT_VERSION
            return result, token_metrics
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
            last_error = str(error)
        except Exception as error:
            last_error = str(error)
            if not last_raw_text:
                last_raw_text = str(error)

        if attempt < config.LLM_RETRIES:
            prompt = _extraction_prompt(context_snippet, last_error)

    return {
        "extraction_failed": True,
        "raw_text": last_raw_text,
        "error": last_error,
        "prompt_version": PROMPT_VERSION,
    }, token_metrics


# --------------------------------------------------------------------------
# 5. BUSINESS RULE VALIDATION
# --------------------------------------------------------------------------

_EVIDENCE_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?")
_ACCEPTANCE_RE = re.compile(
    r"\b(?:closed|close|confirmed|done|final|ho\s+gaya|done\s+hai)\b",
    re.IGNORECASE,
)
_COUNTER_RE = re.compile(
    r"\b(?:counter|revise|revised|instead|at\s+\d|@\s*\d|bid|offer)\b",
    re.IGNORECASE,
)


def _evidence_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_EVIDENCE_DIGITS)
    normalized = normalized.casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _evidence_body(line: str) -> str:
    return re.sub(
        r"^\s*#\d+\s+\[\d{2}:\d{2}:\d{2}\]\s*",
        "",
        line,
    )


def _number_tokens(value: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", value).translate(_EVIDENCE_DIGITS)
    return [
        token.replace(",", "").rstrip(".")
        for token in _NUMBER_TOKEN_RE.findall(normalized)
    ]


def _same_number(left: str, right: str) -> bool:
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def _numbered_lines(context_snippet: Optional[str]) -> Dict[str, str]:
    """Map '#n' to the full message text, including indented continuation lines.
    Quoted 'replying to:' lines are excluded: they belong to another message."""
    if not context_snippet:
        return {}
    result: Dict[str, str] = {}
    current = None
    for line in context_snippet.splitlines():
        header = re.match(r"^\s*(#\d+)\s+\[[^\]]+\]\s+(.*)$", line)
        if header and not line.startswith("    "):
            current = header.group(1)
            result[current] = line.strip()
        elif current and line.startswith("    "):
            if line.strip().lower().startswith("replying to:"):
                continue
            result[current] += "\n" + line.strip()
        else:
            current = None
    return result


def _field_evidence_lines(
    field: str,
    value: Any,
    evidence: Any,
    numbered_lines: Dict[str, str],
) -> List[str]:
    if value is None or not isinstance(evidence, dict):
        return []
    citations = evidence.get(field)
    if not isinstance(citations, list):
        return []
    return [
        numbered_lines[citation]
        for citation in citations
        if isinstance(citation, str) and citation in numbered_lines
    ]


def _value_appears_in_lines(field: str, value: Any, lines: List[str]) -> bool:
    if not lines or value is None:
        return False
    bodies = [_evidence_body(line) for line in lines]
    if field == "isin_text":
        return any(str(value) in line for line in bodies)
    if field in {"ytm", "quantity_text", "coupon_rate", "price"}:
        wanted_numbers = _number_tokens(str(value))
        if not wanted_numbers:
            return False
        for line in bodies:
            numeric_line = re.sub(r"\bT\s*\+\s*\d+\b", " ", line, flags=re.I)
            numeric_line = re.sub(
                r"\b[A-Z]{2}[A-Z0-9]{10}\b",
                " ",
                numeric_line,
                flags=re.I,
            )
            line_numbers = _number_tokens(numeric_line)
            if all(
                any(_same_number(number, found) for found in line_numbers)
                for number in wanted_numbers
            ):
                if field != "quantity_text":
                    return True
                quantity_words = re.findall(
                    r"\b(?:crs?|crores?|lacs?|lakhs?|l|units?)\b",
                    _evidence_normalize(str(value)),
                )
                line_text = _evidence_normalize(line)
                if all(word in line_text.split() for word in quantity_words):
                    return True
        return False
    if field == "owner_side":
        speaker_lines = [
            (line, side)
            for line in lines
            if (side := _speaker_trade_side(_evidence_body(line))) is not None
        ]
        if not speaker_lines:
            return value == "UNKNOWN"
        if value == "UNKNOWN":
            return False
        for line, speaker_side in speaker_lines:
            lower = _evidence_body(line).casefold()
            owner_is_speaker = "[own]" in lower
            owner_side = speaker_side if owner_is_speaker else (
                "SELL" if speaker_side == "BUY" else "BUY"
            )
            if owner_side == value:
                return True
        return False
    if field == "ytm_is_two_way":
        has_two_way_quote = any(
            re.search(r"\b\d{1,2}\.\d+\s*[-–]\s*\d+\b", line)
            for line in bodies
        )
        return has_two_way_quote == value
    if field == "is_indicative":
        indicative = any(
            re.search(r"\b(?:indicative|rfq|inquiry)\b", line, re.I)
            for line in bodies
        )
        firm = any(
            re.search(r"\b(?:firm|close|closed|confirmed|done)\b", line, re.I)
            for line in bodies
        )
        if value:
            return indicative
        return firm
    if field == "status":
        if value == "CONCLUDED":
            return any(_ACCEPTANCE_RE.search(line) for line in bodies)
        if value == "CANCELLED":
            return any(
                _contains_lexicon_entry(line, "pass_or_refusal")
                for line in bodies
            )
        if value == "NEGOTIATING":
            return any(_COUNTER_RE.search(line) for line in bodies)
        if value == "DISCARDED":
            return any(re.search(r"\b(?:payment|allocation|validation|reported|accepted)\b", line, re.I) for line in bodies)
        return bool(lines)
    if field == "evidence":
        return bool(lines)
    normalized_value = _evidence_normalize(str(value))
    return any(
        normalized_value
        and normalized_value in _evidence_normalize(line)
        for line in bodies
    )


def _contains_lexicon_entry(text: str, category: str) -> bool:
    normalized_text = _normalize_lexicon_text(text)
    for entry in LEXICON.get(category, []):
        normalized_entry = _normalize_lexicon_text(entry)
        if normalized_entry and re.search(
            rf"(?<!\w){re.escape(normalized_entry)}(?:s|ed|ing)?(?!\w)",
            normalized_text,
        ):
            return True
    return False


def _speaker_trade_side(text: str) -> Optional[str]:
    if _contains_lexicon_entry(text, "buyer_request_cues"):
        return "BUY"
    if _contains_lexicon_entry(text, "seller_request_cues"):
        return "SELL"
    if _contains_lexicon_entry(text, "seller_offer_cues") or re.search(
        r"(?<!\w)\d+(?:[.,]\d+)?\s*%\s*offers?\b",
        unicodedata.normalize("NFKC", text).translate(_EVIDENCE_DIGITS),
        re.IGNORECASE,
    ):
        return "SELL"
    if _contains_lexicon_entry(text, "buyer_bid_cues"):
        return "BUY"
    if _contains_lexicon_entry(text, "buy_cues"):
        return "BUY"
    if _contains_lexicon_entry(text, "sell_cues"):
        return "SELL"
    return None


def _cancellation_has_linked_offer(
    status_evidence_lines: List[str],
    context_snippet: Optional[str],
) -> bool:
    messages: Dict[int, Dict[str, Any]] = {}
    current_number: Optional[int] = None
    for line in (context_snippet or "").splitlines():
        header = re.match(r"^\s*#(\d+)\s+\[[^\]]+\]\s+(.*)$", line)
        if header and not line.startswith("    "):
            current_number = int(header.group(1))
            message_match = re.match(
                r"^\s*(?:\[OWN\]\s*)?(?P<sender>[^:]+):\s*(?P<body>.*)$",
                header.group(2),
            )
            messages[current_number] = {
                "sender": (
                    message_match.group("sender").strip().casefold()
                    if message_match
                    else None
                ),
                "body": message_match.group("body") if message_match else "",
                "replying_to": None,
            }
        elif current_number is not None and line.startswith("    "):
            continuation = line.strip()
            if continuation.casefold().startswith("replying to:"):
                messages[current_number]["replying_to"] = continuation[
                    len("replying to:"):
                ].strip()
            else:
                messages[current_number]["body"] += "\n" + continuation
        else:
            current_number = None

    offer_categories = ("buy_cues", "sell_cues", "question_cues")

    def is_offer_or_inquiry(message: Dict[str, Any]) -> bool:
        return any(
            _contains_lexicon_entry(message["body"], category)
            for category in offer_categories
        )

    refusal_numbers = {
        int(match.group(1))
        for line in status_evidence_lines
        if (match := re.match(r"#(\d+)", line)) is not None
        and _contains_lexicon_entry(
            _evidence_body(line),
            "pass_or_refusal",
        )
    }
    for refusal_number in refusal_numbers:
        refusal = messages.get(refusal_number)
        if refusal is None or refusal["sender"] is None:
            continue
        possible_targets = []
        previous = messages.get(refusal_number - 1)
        if previous is not None:
            possible_targets.append(previous)
        quote = refusal["replying_to"]
        if quote:
            normalized_quote = _normalize_lexicon_text(quote)
            possible_targets.extend(
                message
                for number, message in messages.items()
                if number < refusal_number
                and normalized_quote
                and normalized_quote
                in _normalize_lexicon_text(message["body"])
            )
        for target in possible_targets:
            if (
                is_offer_or_inquiry(target)
                and target["sender"] is not None
                and target["sender"] != refusal["sender"]
            ):
                return True
    return False


def _evidence_check(
    deal: Dict[str, Any],
    context_snippet: Optional[str],
) -> Tuple[Dict[str, bool], List[str], Dict[str, List[str]]]:
    evidence = deal.get("evidence") or {}
    lines = _numbered_lines(context_snippet)
    verified: Dict[str, bool] = {}
    warnings: List[str] = []
    cited_lines: Dict[str, List[str]] = {}

    for field, value in list(deal.items()):
        if field in {"evidence", "status_evidence", "partial_isin_text"} or value is None:
            continue
        selected_lines = _field_evidence_lines(field, value, evidence, lines)
        cited_lines[field] = selected_lines
        is_verified = _value_appears_in_lines(field, value, selected_lines)
        verified[field] = is_verified
        if not is_verified:
            deal[field] = None
            warnings.append(f"unsupported_{field}")

    status_evidence = deal.get("status_evidence")
    if status_evidence is not None:
        selected_lines = _field_evidence_lines(
            "status_evidence",
            status_evidence,
            evidence,
            lines,
        )
        cited_lines["status_evidence"] = selected_lines
        verified["status_evidence"] = _value_appears_in_lines(
            "status_evidence",
            status_evidence,
            selected_lines,
        )
        if not verified["status_evidence"]:
            deal["status_evidence"] = None
            warnings.append("unsupported_status_evidence")

    return verified, warnings, cited_lines


def _correct_owner_side_from_cited_lines(
    deal: Dict[str, Any],
    context_snippet: Optional[str],
    warnings: List[str],
) -> None:
    evidence = deal.get("evidence")
    if not isinstance(evidence, dict):
        return
    numbered_lines = _numbered_lines(context_snippet)
    citations = {
        citation
        for field_citations in evidence.values()
        if isinstance(field_citations, list)
        for citation in field_citations
        if isinstance(citation, str)
    }
    cited_lines = [
        line for citation, line in numbered_lines.items() if citation in citations
    ]
    if not cited_lines:
        return

    owner_lines = [
        _evidence_body(line)
        for line in cited_lines
        if re.match(r"^\s*\[OWN\](?:\s|$)", _evidence_body(line), re.IGNORECASE)
    ]
    bodies = [_evidence_body(line) for line in cited_lines]
    has_offer_level = any(
        re.search(
            r"\b(?:"
            r"\d+(?:[.,]\d+)?\s*%?\s*offers?"
            r"|offers?\s*(?:pls|please)?\s*(?:at|@)\s*\d"
            r")",
            body,
            re.IGNORECASE,
        )
        for body in bodies
    )
    correction: Optional[str] = None
    if owner_lines and not has_offer_level and any(
        _contains_lexicon_entry(line, "buyer_request_cues")
        for line in owner_lines
    ):
        correction = "BUY"
    elif owner_lines and any(
        _contains_lexicon_entry(line, "seller_request_cues")
        for line in owner_lines
    ):
        correction = "SELL"

    has_isin_with_units = any(
        re.search(r"\b[A-Z]{2}[A-Z0-9]{10}\b", body, re.IGNORECASE)
        and (
            re.search(r"\b\d+\s*(?:units?|pieces?|pcs?)\b", body, re.IGNORECASE)
            or re.search(
                r"\b[A-Z]{2}[A-Z0-9]{10}\s*[-–]\s*\d+\b",
                body,
                re.IGNORECASE,
            )
        )
        for body in bodies
    )
    has_side_cue = any(_speaker_trade_side(body) is not None for body in bodies)
    if correction is not None:
        deal["owner_side"] = correction
    elif has_isin_with_units and not has_side_cue:
        correction = "UNKNOWN"
        deal["owner_side"] = correction
        deal["buyer_text"] = None
        deal["seller_text"] = None

    if correction is not None:
        warnings[:] = [
            warning for warning in warnings if warning != "unsupported_owner_side"
        ]
        if "side_corrected" not in warnings:
            warnings.append("side_corrected")


def _settlement_reference_datetime(
    context_snippet: Optional[str],
    settlement_citations: List[str],
    fallback: Optional[datetime],
) -> Optional[datetime]:
    date_match = re.search(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s*$", context_snippet or "", re.M)
    if date_match:
        for line in (context_snippet or "").splitlines():
            for citation in settlement_citations:
                time_match = re.match(
                    rf"^{re.escape(citation)}\s+\[(\d{{2}}:\d{{2}}:\d{{2}})\]",
                    line,
                )
                if not time_match:
                    continue
                try:
                    return datetime.fromisoformat(
                        f"{date_match.group(1)}T{time_match.group(1)}"
                    )
                except ValueError:
                    return fallback
    return fallback


def _normalize_extracted_fields(
    deal: Dict[str, Any],
    context_snippet: Optional[str],
    message_datetime: Optional[datetime],
    cited_lines: Dict[str, List[str]],
) -> Tuple[List[str], bool]:
    warnings = []
    needs_review = False

    quantity_text = deal.get("quantity_text")
    quantity_input = quantity_text
    if quantity_text is not None and normalize_quantity(quantity_text) is None:
        quantity_lines = cited_lines.get("quantity_text", [])
        isin_match = next(
            (
                match.group(0)
                for line in quantity_lines
                if (match := re.search(r"\b[A-Z]{2}[A-Z0-9]{10}\b", line, re.I))
            ),
            None,
        )
        if isin_match:
            quantity_input = f"{quantity_text} {isin_match}"
    deal["normalized_quantity"] = (
        normalize_quantity(quantity_input)
        if quantity_text is not None
        else None
    )
    if quantity_text is not None and deal["normalized_quantity"] is None:
        warnings.append("invalid_quantity_text")
        needs_review = True

    maturity_text = deal.get("maturity_text")
    maturity = (
        normalize_maturity(maturity_text)
        if maturity_text is not None
        else {"maturity_text": None, "maturity_date": None, "maturity_ym": None}
    )
    deal["maturity_date"] = maturity["maturity_date"]
    deal["maturity_ym"] = maturity["maturity_ym"]
    if maturity_text is not None and maturity["maturity_date"] is None and maturity["maturity_ym"] is None:
        warnings.append("unparsed_maturity_text")
        needs_review = True

    ytm_text = deal.get("ytm")
    if ytm_text is not None:
        clean_ytm = str(ytm_text).replace("%", "").strip()
        parsed_number = None
        try:
            parsed_number = float(clean_ytm)
        except ValueError:
            pass

        yield_result = normalize_yield_coupon(f"YTM {ytm_text}")
        deal["ytm_values"] = yield_result["yields"]
        if yield_result["two_way_quotes"]:
            deal["ytm_is_two_way"] = True

        is_out_of_range = False
        if parsed_number is not None:
            if not (3.0 <= parsed_number <= 25.0):
                is_out_of_range = True
        elif yield_result["yields"]:
            if any(not (3.0 <= y <= 25.0) for y in yield_result["yields"]):
                is_out_of_range = True
        elif yield_result["warnings"]:
            is_out_of_range = True

        if is_out_of_range:
            deal["ytm"] = None
            deal["ytm_values"] = []
            if "yield_out_of_expected_range" not in warnings:
                warnings.append("yield_out_of_expected_range")
            needs_review = True
        else:
            if parsed_number is not None:
                deal["ytm"] = parsed_number
            elif yield_result["yields"] and not yield_result["two_way_quotes"]:
                deal["ytm"] = yield_result["yields"][0]
            elif clean_ytm:
                deal["ytm"] = clean_ytm
            else:
                deal["ytm"] = None
    else:
        deal["ytm"] = None
        deal["ytm_values"] = []

    settlement_text = deal.get("settlement_text")
    settlement_citations = (
        deal.get("evidence", {}).get("settlement_text", [])
        if isinstance(deal.get("evidence"), dict)
        else []
    )
    settlement_datetime = _settlement_reference_datetime(
        context_snippet,
        settlement_citations if isinstance(settlement_citations, list) else [],
        message_datetime,
    )
    settlement = None
    if settlement_text is not None:
        if settlement_datetime is None:
            settlement = {
                "settlement_date": None,
                "label": "OTHER",
                "needs_review": True,
            }
        else:
            settlement = resolve_settlement(
                settlement_text,
                settlement_datetime,
                market_holidays=config.MARKET_HOLIDAYS,
            )
        needs_review = needs_review or settlement["needs_review"]
    deal["settlement_date"] = settlement["settlement_date"] if settlement else None
    deal["settlement_label"] = settlement["label"] if settlement else None
    deal["settlement_needs_review"] = (
        settlement["needs_review"] if settlement else False
    )
    if deal["settlement_needs_review"]:
        warnings.append("settlement_needs_review")

    return warnings, needs_review


def _conclusion_check(
    deal: Dict[str, Any],
    verified: Dict[str, bool],
    cited_lines: Dict[str, List[str]],
) -> List[str]:
    missing = []
    identity_verified = any(
        deal.get(field) is not None and verified.get(field, False)
        for field in ("bond_identity_text", "isin_text")
    )
    if not identity_verified:
        missing.append("bond_identity_text_or_isin_text")
    if deal.get("ytm") is None or not verified.get("ytm", False):
        missing.append("ytm")
    if deal.get("quantity_text") is None or not verified.get("quantity_text", False):
        missing.append("quantity_text")
    side_present = any(
        deal.get(field) is not None and verified.get(field, False)
        for field in ("owner_side", "buyer_text", "seller_text")
    )
    if not side_present:
        missing.append("owner_side_or_buyer_text_or_seller_text")

    term_fields = ["ytm", "price"]
    term_line_numbers = [
        int(match.group(1))
        for field in term_fields
        if deal.get(field) is not None and verified.get(field, False)
        for line in cited_lines.get(field, [])
        if (match := re.match(r"#(\d+)", line)) is not None
    ]
    acceptance_lines = cited_lines.get("status_evidence", [])
    acceptance_numbered = [
        (int(match.group(1)), line)
        for line in acceptance_lines
        if (match := re.match(r"#(\d+)", line)) is not None
    ]
    has_later_acceptance = any(
        number > min(term_line_numbers, default=0) and _ACCEPTANCE_RE.search(line)
        for number, line in acceptance_numbered
    )
    if not has_later_acceptance:
        missing.append("later_acceptance_evidence")

    return missing


def validate_extraction(
    deal: Any,
    context_snippet: Optional[str] = None,
    message_datetime: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Validate, normalize, and evidence-check a single extracted deal."""
    if not isinstance(deal, dict):
        return {
            "extraction_failed": True,
            "raw_text": json.dumps(deal, ensure_ascii=False, default=str),
            "error": "Extraction output must be a deal object",
            "prompt_version": PROMPT_VERSION,
        }
    status_missing_defaulted = deal.get("status") not in EXTRACTION_STATUSES
    deal_to_validate = dict(deal)
    if status_missing_defaulted:
        deal_to_validate["status"] = "UNCERTAIN"
    try:
        validated = DealExtraction.model_validate(
            deal_to_validate
        ).model_dump(mode="json")
    except ValidationError as error:
        return {
            "extraction_failed": True,
            "raw_text": json.dumps(deal, ensure_ascii=False, default=str),
            "error": str(error),
            "prompt_version": PROMPT_VERSION,
        }

    errors: List[str] = []
    warnings: List[str] = []
    if status_missing_defaulted:
        warnings.append("status_missing_defaulted")

    isin_text = validated.get("isin_text")
    if isin_text is not None and str(isin_text).strip():
        raw_isin = str(isin_text).strip()
        if not is_valid_isin(raw_isin):
            validated["partial_isin_text"] = isin_text
            validated["isin_text"] = None
            warnings.append("partial_isin")
            if isinstance(validated.get("evidence"), dict) and "isin_text" in validated["evidence"]:
                validated["evidence"]["partial_isin_text"] = validated["evidence"].get("isin_text")
        else:
            validated["isin_text"] = raw_isin
            validated["partial_isin_text"] = None
    else:
        validated["isin_text"] = None
        if not validated.get("partial_isin_text"):
            validated["partial_isin_text"] = None

    verified, evidence_warnings, cited_lines = _evidence_check(
        validated,
        context_snippet,
    )
    warnings.extend(evidence_warnings)
    _correct_owner_side_from_cited_lines(validated, context_snippet, warnings)
    if validated.get("status") not in EXTRACTION_STATUSES:
        validated["status"] = "UNCERTAIN"
        if "status_missing_defaulted" not in warnings:
            warnings.append("status_missing_defaulted")
    normalization_warnings, normalizer_needs_review = _normalize_extracted_fields(
        validated,
        context_snippet,
        message_datetime,
        cited_lines,
    )
    warnings.extend(normalization_warnings)

    missing_fields = _conclusion_check(validated, verified, cited_lines)
    original_status = validated.get("status")
    if original_status == "CANCELLED":
        cancellation_linked = (
            verified.get("status", False)
            and verified.get("status_evidence", False)
            and _cancellation_has_linked_offer(
                cited_lines.get("status_evidence", []),
                context_snippet,
            )
        )
        if cancellation_linked:
            validated["status"] = "CANCELLED"
        else:
            validated["status"] = "UNCERTAIN"
            missing_fields.append("linked_offer_or_inquiry")
            warnings.append("unlinked_cancellation_refusal")
    if (
        validated.get("buyer_text")
        and validated.get("seller_text")
        and validated["buyer_text"].casefold() == validated["seller_text"].casefold()
    ):
        errors.append("buyer_equals_seller")
    if original_status == "CONCLUDED" and missing_fields:
        validated["status"] = (
            "NEGOTIATING"
            if any(_COUNTER_RE.search(line) for line in (context_snippet or "").splitlines())
            else "UNCERTAIN"
        )
        warnings.append("conclusion_downgraded")

    key_checks = {
        "bond_identity": any(
            validated.get(field) is not None and verified.get(field, False)
            for field in ("bond_identity_text", "isin_text")
        ),
        "ytm": validated.get("ytm") is not None and verified.get("ytm", False),
        "quantity": (
            validated.get("quantity_text") is not None
            and verified.get("quantity_text", False)
        ),
        "side": any(
            validated.get(field) is not None and verified.get(field, False)
            for field in ("owner_side", "buyer_text", "seller_text")
        ),
    }
    if original_status == "CONCLUDED":
        key_checks["acceptance"] = (
            "later_acceptance_evidence" not in missing_fields
        )
    confidence = sum(key_checks.values()) / len(key_checks)
    validated["errors"] = errors
    validated["warnings"] = warnings
    validated["missing_fields"] = missing_fields
    validated["confidence"] = confidence
    validated["needs_review"] = (
        confidence < config.REVIEW_CONFIDENCE_THRESHOLD
        or normalizer_needs_review
        or validated.get("settlement_needs_review", False)
    )
    return validated
