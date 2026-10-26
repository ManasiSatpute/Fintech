"""Mask personally identifiable details from text sent to the language model."""

import re
from typing import Any, Dict, Iterable, Mapping, Tuple


_PROTECTED_PATTERNS = (
    re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b", re.IGNORECASE),
    re.compile(r"\bR\d{5,}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:19|20|21)\d{2}[-/.]\d{1,2}[-/.]\d{1,4}\b"
        r"|\b\d{1,2}[-/.]\d{1,2}[-/.](?:\d{2}|\d{4})\b"
        r"|\b(?:19|20|21)\d{2}\b"
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|bps?\b|"
        r"(?:cr(?:ore)?s?|l(?:akh|ac)?s?|units?|qty|million|mn)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:yield|ytm|quantity|qty|units?)\s*[:=]?\s*"
        r"\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?",
        re.IGNORECASE,
    ),
)
_SENSITIVE_PATTERNS = (
    re.compile(
        r"(?<![@\w])(?:https?://|www\.)\S+|"
        r"(?<![@\w])(?:[A-Z0-9-]+\.)+[A-Z]{2,}(?:/\S*)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\w)@\d+\b"),
    re.compile(
        r"\b(?:[A-Z][A-Z0-9&.-]{1,15}(?:\s+bank)?\s+\d{3,8}|"
        r"(?:account|acct|a/c)\s*(?:no\.?\s*)?[:#-]?\s*\d{2,12})\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![A-Z0-9])\+?\d(?:[\d ()-]{5,}\d)(?![A-Z0-9])", re.IGNORECASE),
)
_PHONE_DIGITS = re.compile(r"\d")
_DEVICE_SUFFIX = re.compile(r":\d+$")
_TRAILING_URL_PUNCTUATION = ".,!?;:)]}"


def _mask_value(
    value: str,
    placeholder_map: Dict[str, str],
    reverse_map: Dict[str, str],
) -> str:
    if value not in reverse_map:
        index = len(placeholder_map) + 1
        placeholder = f"[PII_{index:03d}]"
        while placeholder in placeholder_map or placeholder in value:
            index += 1
            placeholder = f"[PII_{index:03d}]"
        placeholder_map[placeholder] = value
        reverse_map[value] = placeholder
    return reverse_map[value]


def mask_pii(
    text: str,
    sender_numbers: Iterable[str] = (),
    person_names: Iterable[str] = (),
    mask_person_names: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """Return masked text and a local map that can restore its placeholders."""
    placeholder_map: Dict[str, str] = {}
    reverse_map: Dict[str, str] = {}
    protected: Dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        token = f"\x00KEEP{len(protected)}\x00"
        protected[token] = match.group(0)
        return token

    for pattern in _PROTECTED_PATTERNS:
        text = pattern.sub(protect, text)

    for sender_number in sender_numbers:
        number = str(sender_number).strip()
        if not number:
            continue
        variants = {number}
        device_free_number = _DEVICE_SUFFIX.sub("", number)
        if device_free_number:
            variants.add(device_free_number)
        for variant in sorted(variants, key=len, reverse=True):
            text = re.sub(
                rf"(?<!\w){re.escape(variant)}(?!\w)",
                lambda match: _mask_value(
                    match.group(0), placeholder_map, reverse_map
                ),
                text,
            )

    def mask_pattern(pattern: re.Pattern[str]) -> None:
        nonlocal text

        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if pattern is _SENSITIVE_PATTERNS[0]:
                value = value.rstrip(_TRAILING_URL_PUNCTUATION)
                suffix = match.group(0)[len(value):]
                return _mask_value(value, placeholder_map, reverse_map) + suffix
            if pattern is _SENSITIVE_PATTERNS[-1] and len(
                _PHONE_DIGITS.findall(value)
            ) < 7:
                return value
            return _mask_value(value, placeholder_map, reverse_map)

        text = pattern.sub(replace, text)

    for pattern in _SENSITIVE_PATTERNS:
        mask_pattern(pattern)

    if mask_person_names:
        for name in sorted(
            {str(name).strip() for name in person_names if str(name).strip()},
            key=len,
            reverse=True,
        ):
            text = re.sub(
                rf"(?<!\w){re.escape(name)}(?!\w)",
                lambda match: _mask_value(
                    match.group(0), placeholder_map, reverse_map
                ),
                text,
                flags=re.IGNORECASE,
            )

    for token, original in protected.items():
        text = text.replace(token, original)
    return text, placeholder_map


def restore_pii(value: Any, placeholder_map: Mapping[str, str]) -> Any:
    """Restore placeholders in model output while preserving its data shape."""
    if isinstance(value, str):
        if not placeholder_map:
            return value
        placeholders = re.compile(
            "|".join(re.escape(key) for key in placeholder_map)
        )
        return placeholders.sub(
            lambda match: placeholder_map[match.group(0)],
            value,
        )
    if isinstance(value, list):
        return [restore_pii(item, placeholder_map) for item in value]
    if isinstance(value, dict):
        return {
            key: restore_pii(item, placeholder_map)
            for key, item in value.items()
        }
    return value
