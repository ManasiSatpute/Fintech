"""Environment-overridable application and pipeline settings."""

import os
from datetime import date
from pathlib import Path
from typing import List, Optional


def _list_setting(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_setting(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _int_setting(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    result = default if value is None else int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


OWN_SENDER_NAMES = _list_setting("OWN_SENDER_NAMES", [])
SENDER_HASH_SALT = "fintech-sender-id-hash-v1"
DAY_FIRST_DEFAULT = _bool_setting("DAY_FIRST_DEFAULT", True)
GAP_MINUTES = _int_setting("GAP_MINUTES", 15)

_max_contexts = os.getenv("MAX_CONTEXTS", "none").strip().lower()
MAX_CONTEXTS: Optional[int] = (
    None if _max_contexts == "none" else int(_max_contexts)
)
if MAX_CONTEXTS is not None and MAX_CONTEXTS < 1:
    raise ValueError("MAX_CONTEXTS must be at least 1 or 'none'")
UPLOAD_SESSION_TTL_SECONDS = _int_setting(
    "UPLOAD_SESSION_TTL_SECONDS",
    3600,
    minimum=1,
)

MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:31b-cloud")
LLM_RETRIES = _int_setting("LLM_RETRIES", 1)

MARKET_HOLIDAYS = _list_setting("MARKET_HOLIDAYS", [])
for holiday in MARKET_HOLIDAYS:
    date.fromisoformat(holiday)

PII_MASKING_ENABLED = _bool_setting("PII_MASKING_ENABLED", True)
MASK_PERSON_NAMES = _bool_setting("MASK_PERSON_NAMES", False)
RELEVANCE_THRESHOLD = _int_setting("RELEVANCE_THRESHOLD", 3)
REVIEW_CONFIDENCE_THRESHOLD = float(
    os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "0.7")
)
if not 0.0 <= REVIEW_CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError("REVIEW_CONFIDENCE_THRESHOLD must be between 0 and 1")

# Dynamic Deal Stacking Thresholds
STACK_HIGH_CONFIDENCE_THRESHOLD = float(
    os.getenv("STACK_HIGH_CONFIDENCE_THRESHOLD", "0.65")
)
if not 0.0 <= STACK_HIGH_CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError("STACK_HIGH_CONFIDENCE_THRESHOLD must be between 0 and 1")

STACK_NEW_NEGOTIATION_THRESHOLD = float(
    os.getenv("STACK_NEW_NEGOTIATION_THRESHOLD", "0.30")
)
if not 0.0 <= STACK_NEW_NEGOTIATION_THRESHOLD <= 1.0:
    raise ValueError("STACK_NEW_NEGOTIATION_THRESHOLD must be between 0 and 1")

STACK_AMBIGUITY_DELTA_THRESHOLD = float(
    os.getenv("STACK_AMBIGUITY_DELTA_THRESHOLD", "0.15")
)
if not 0.0 <= STACK_AMBIGUITY_DELTA_THRESHOLD <= 1.0:
    raise ValueError("STACK_AMBIGUITY_DELTA_THRESHOLD must be between 0 and 1")

STACK_TIME_DECAY_HOURS = float(
    os.getenv("STACK_TIME_DECAY_HOURS", "4.0")
)
if STACK_TIME_DECAY_HOURS <= 0.0:
    raise ValueError("STACK_TIME_DECAY_HOURS must be greater than 0")

ALLOWED_ORIGINS = _list_setting(
    "ALLOWED_ORIGINS",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
)
MODEL_CONTEXT_LENGTH_FALLBACK = _int_setting(
    "MODEL_CONTEXT_LENGTH_FALLBACK", 262144, minimum=1
)

BOND_REFERENCE_CSV = Path(
    os.getenv(
        "BOND_REFERENCE_CSV",
        str(Path(__file__).resolve().parent / "ISIN_DETAILS.csv"),
    )
)
