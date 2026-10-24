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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama
from langchain_ollama import ChatOllama

import config

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
        identity = json.dumps(
            [
                record["chat_key"],
                record["timestamp"],
                record["sender"],
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
            "snippet_text": format_context_snippet(burst_messages, group_name),
            "relevance_score": relevance_score,
        }
        if relevance_score >= config.RELEVANCE_THRESHOLD:
            contexts.append(context)
        else:
            skipped_low_relevance.append(context)

    if return_skipped:
        return contexts, skipped_low_relevance
    return contexts


def format_context_snippet(window: List[Dict[str, Any]], group_name: str) -> str:
    lines = []
    if group_name and group_name != "Default":
        lines.append(f"Channel: {group_name}")
    for m in window:
        time_str = m["dt"].strftime("%H:%M:%S")
        text = "[attachment]" if m.get("is_media", False) else m["message_text"]
        lines.append(f"[{time_str}] {m['sender']}: {text}")
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
    return config.MODEL_CONTEXT_LENGTH_FALLBACK


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
