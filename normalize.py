"""Pure deterministic normalizers for bond-trade values."""

import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple


_DIGIT_TRANSLATION = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER = r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?"
_QUANTITY_UNIT = (
    r"crs?|crores?|lacs?|lakhs?|l|units?"
)
_QUANTITY_RE = re.compile(
    rf"(?P<only>\bonly\b\s*)?"
    rf"(?:(?P<multiple>\bmultiple\s+of\b)\s*)?"
    rf"(?:(?P<entire>\bentire\s+quantum\b)|"
    rf"(?P<min>{_NUMBER})(?:\s*(?:-|–|to)\s*(?P<max>{_NUMBER}))?"
    rf"\s*(?P<unit>{_QUANTITY_UNIT})?)"
    rf"(?P<only_after>\s+\bonly\b)?",
    re.IGNORECASE,
)
_ISIN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}[A-Z0-9]{10}(?![A-Z0-9])", re.I)
_PARTIAL_ISIN_RE = re.compile(
    r"\bISIN\b\s*[:#-]?\s*(?:[A-Z]{2}\s*)?(?P<fragment>\d{1,11})\b",
    re.IGNORECASE,
)
_BPS_RE = re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*bps?\b", re.I)
_TWO_WAY_RE = re.compile(
    r"(?<![\d.])(?P<first>\d{1,2}\.\d{1,4})\s*[-–]\s*(?P<second>\d{1,4})(?!\d)"
)
_PERCENT_RE = re.compile(r"(?<![\d.])(?P<value>\d{1,2}(?:\.\d+)?)\s*%")
_UNMARKED_YIELD_RE = re.compile(
    r"(?<![\d.])(?P<value>\d{1,2}\.\d{1,4})(?!\d)"
)
_YIELD_CUE = re.compile(
    r"(?:@|(?:\bbid\b|\boffer\b|\blvl\b|\blevel\b|\bbest\b|\bytm\b|"
    r"\byield\b|\bcan\s+take\b))\s*[:=]?\s*$",
    re.I,
)
_YIELD_CUE_AFTER = re.compile(
    r"^\s*(?:bid|offer|lvl|level|best|ytm|yield|can\s+take)\b",
    re.I,
)
_MONTH_ALIASES = {
    alias: month
    for month, name in enumerate(
        (
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
    )
    if name
    for alias in (name.casefold(), name[:3].casefold())
}
_MATURITY_PATTERNS = [
    re.compile(
        r"\b(?P<day>\d{1,2})\s+"
        r"(?P<month>[A-Za-z]+)\s+(?P<year>\d{2,4})\b",
        re.I,
    ),
    re.compile(
        r"\b(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})"
        r"(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{2,4})\b",
        re.I,
    ),
    re.compile(
        r"\b(?P<month>[A-Za-z]+)\s+(?P<year>\d{2,4})\b",
        re.I,
    ),
]
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).translate(_DIGIT_TRANSLATION)


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _unit_scale(unit: Optional[str]) -> Tuple[str, Optional[float]]:
    if not unit:
        return "", None
    normalized = unit.casefold()
    if normalized in {"cr", "crs", "crore", "crores"}:
        return "Cr", 10_000_000
    if normalized in {"l", "lac", "lacs", "lakh", "lakhs"}:
        return "Lakh", 100_000
    return "Units", None


def normalize_quantity(
    text: str,
    face_value: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Extract the first explicit quantity and report its normalized amount."""
    normalized_text = _normalize_text(text)
    match = _QUANTITY_RE.search(normalized_text)
    if not match:
        multiple_match = re.search(r"\bmultiple\s+of\b", normalized_text, re.I)
        if not multiple_match:
            return None
        return {
            "as_said": multiple_match.group(0),
            "value": None,
            "unit": None,
            "amount_inr": None,
            "units_assumed": False,
            "is_indicative": False,
            "only": False,
            "multiple_of": True,
            "entire_quantum": False,
        }

    as_said = match.group(0).strip()
    unit, scale = _unit_scale(match.group("unit"))
    min_value = _number(match.group("min")) if match.group("min") else None
    max_value = _number(match.group("max")) if match.group("max") else None
    is_range = max_value is not None
    is_units = unit == "Units"
    units_assumed = False

    if unit == "" and min_value is not None:
        isin_match = _ISIN_RE.search(normalized_text)
        number_start = match.start("min")
        number_end = match.end("min")
        near_isin = isin_match is not None and (
            abs(number_start - isin_match.start()) <= 40
            or abs(number_end - isin_match.end()) <= 40
        )
        if near_isin:
            unit = "Units"
            is_units = True
            units_assumed = True
        elif not match.group("multiple"):
            return None

    if is_units:
        multiplier = face_value if face_value is not None else None
        amount_min = (
            min_value * multiplier
            if min_value is not None and multiplier is not None
            else None
        )
        amount_max = (
            max_value * multiplier
            if max_value is not None and multiplier is not None
            else None
        )
    elif scale is not None:
        amount_min = min_value * scale if min_value is not None else None
        amount_max = max_value * scale if max_value is not None else None
    else:
        amount_min = None
        amount_max = None

    value: Any = (
        {"min": min_value, "max": max_value}
        if is_range
        else min_value
    )
    result = {
        "as_said": as_said,
        "value": value,
        "unit": unit or None,
        "amount_inr": (
            {"min": amount_min, "max": amount_max}
            if is_range
            else amount_min
        ),
        "units_assumed": units_assumed,
        "is_indicative": is_range,
        "only": bool(match.group("only") or match.group("only_after")),
        "multiple_of": bool(match.group("multiple")),
        "entire_quantum": bool(match.group("entire")),
    }
    return result


def normalize_yield_coupon(
    text: str,
    issuer_aliases: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Separate coupon rates, yield quotes, and basis-point spreads."""
    normalized_text = _normalize_text(text)
    issuer_names = [name for name in (issuer_aliases or []) if name]
    issuer_pattern = (
        re.compile("|".join(re.escape(name) for name in issuer_names), re.I)
        if issuer_names
        else None
    )

    coupons = []
    yields = []
    two_way_quotes = []
    spreads = []
    for match in _BPS_RE.finditer(normalized_text):
        value = _number(match.group("value"))
        spreads.append({
            "as_said": match.group(0),
            "bps": value,
            "percent": value / 100,
        })

    for match in _TWO_WAY_RE.finditer(normalized_text):
        first_text = match.group("first")
        suffix = match.group("second")
        integer_part, decimal_part = first_text.split(".", 1)
        if len(suffix) <= len(decimal_part):
            second_text = f"{integer_part}.{suffix.zfill(len(decimal_part))}"
        else:
            second_text = suffix
        first_value = _number(first_text)
        second_value = _number(second_text)
        quote = {
            "as_said": match.group(0),
            "values": [first_value, second_value],
        }
        two_way_quotes.append(quote)
        yields.extend([first_value, second_value])

    quote_spans = [match.span() for match in _TWO_WAY_RE.finditer(normalized_text)]
    for match in _PERCENT_RE.finditer(normalized_text):
        if any(start <= match.start() < end for start, end in quote_spans):
            continue
        value = _number(match.group("value"))
        before = normalized_text[max(0, match.start() - 45):match.start()]
        after = normalized_text[match.end():match.end() + 80]
        issuer_after = issuer_pattern.search(after) if issuer_pattern else None
        if issuer_after:
            coupons.append({
                "as_said": match.group(0),
                "value": value,
                "issuer": issuer_after.group(0),
            })
        elif _YIELD_CUE.search(before) or _YIELD_CUE_AFTER.match(after):
            yields.append(value)

    for match in _UNMARKED_YIELD_RE.finditer(normalized_text):
        if any(start <= match.start() < end for start, end in quote_spans):
            continue
        if any(start <= match.start() < end for start, end in (m.span() for m in _PERCENT_RE.finditer(normalized_text))):
            continue
        before = normalized_text[max(0, match.start() - 45):match.start()]
        after = normalized_text[match.end():match.end() + 35]
        if _YIELD_CUE.search(before) or _YIELD_CUE_AFTER.match(after):
            yields.append(_number(match.group("value")))

    unique_yields = list(dict.fromkeys(yields))
    warnings = [
        f"Yield {value:g}% is outside the expected 3%-25% range"
        for value in unique_yields
        if not 3 <= value <= 25
    ]
    return {
        "coupons": coupons,
        "yields": unique_yields,
        "two_way_quotes": two_way_quotes,
        "ytm_is_two_way": bool(two_way_quotes),
        "spreads": spreads,
        "warnings": warnings,
    }


def _is_business_day(value: date, holidays: set) -> bool:
    return value.weekday() < 5 and value not in holidays


def _next_business_day(value: date, holidays: set, include_today: bool = False) -> date:
    candidate = value if include_today else value + timedelta(days=1)
    while not _is_business_day(candidate, holidays):
        candidate += timedelta(days=1)
    return candidate


def _add_business_days(value: date, count: int, holidays: set) -> date:
    if count == 0:
        return _next_business_day(value, holidays, include_today=True)
    result = value
    for _ in range(count):
        result = _next_business_day(result, holidays)
    return result


def _next_weekday(value: date, weekday: int) -> date:
    days_ahead = (weekday - value.weekday()) % 7
    return value + timedelta(days=days_ahead or 7)


def resolve_settlement(
    settlement_text: str,
    message_datetime: datetime,
    market_holidays: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Resolve explicit settlement hints without inventing a default date."""
    try:
        from config import MARKET_HOLIDAYS
    except ImportError:
        MARKET_HOLIDAYS = []
    holidays = {
        date.fromisoformat(value)
        for value in (market_holidays if market_holidays is not None else MARKET_HOLIDAYS)
    }
    normalized = _normalize_text(settlement_text).casefold()
    base_date = message_datetime.date()
    t_plus_match = re.search(r"\bt\s*\+\s*(\d+)\b|\bt([012])\b", normalized)
    t_plus = (
        int(t_plus_match.group(1) or t_plus_match.group(2))
        if t_plus_match
        else None
    )

    relative_date = None
    relative_label = None
    if re.search(r"\b(?:tom|tomorrow)\b", normalized):
        relative_date = base_date + timedelta(days=1)
        relative_label = "tomorrow"
    elif re.search(r"\b(?:today|aaj)\b", normalized):
        relative_date = base_date
        relative_label = "today"
    elif re.search(r"\bkal\b", normalized):
        relative_date = base_date + timedelta(days=1)
        relative_label = "kal"
    elif re.search(r"\bparso\b", normalized):
        relative_date = base_date + timedelta(days=2)
        relative_label = "parso"

    weekday_date = None
    weekday_label = None
    weekday_aliases = {
        "mon": "monday",
        "tue": "tuesday",
        "tues": "tuesday",
        "wed": "wednesday",
        "thu": "thursday",
        "thur": "thursday",
        "thurs": "thursday",
        "fri": "friday",
        "sat": "saturday",
        "sun": "sunday",
    }
    for name, weekday in _WEEKDAYS.items():
        aliases = [name, *(alias for alias, full_name in weekday_aliases.items() if full_name == name)]
        if any(re.search(rf"\b{alias}\b", normalized) for alias in aliases):
            weekday_date = _next_weekday(base_date, weekday)
            weekday_label = name
            break
    if weekday_date is not None and re.search(r"\bvd\b|\bvalue date\b", normalized):
        relative_date = weekday_date
        relative_label = f"{weekday_label} VD"

    tom_t0 = relative_label == "tomorrow" and t_plus == 0
    ambiguous = (
        t_plus is not None
        and relative_date is not None
        and not tom_t0
    )
    t_plus_date = (
        _add_business_days(base_date, t_plus, holidays)
        if t_plus is not None
        else None
    )
    if tom_t0:
        t_plus_date = _next_business_day(base_date, holidays)
    resolved_date = (
        t_plus_date if t_plus_date is not None else relative_date
    )
    if resolved_date is None and weekday_date is not None:
        resolved_date = weekday_date

    label = f"T{t_plus}" if t_plus is not None and t_plus in {0, 1, 2} else "OTHER"
    if t_plus is None and resolved_date is not None:
        label = "OTHER"

    if tom_t0:
        resolved_date = _next_business_day(base_date, holidays)

    return {
        "settlement_date": resolved_date.isoformat() if resolved_date else None,
        "label": label,
        "needs_review": ambiguous,
        "relative_part": (
            {"text": relative_label, "date": relative_date.isoformat()}
            if relative_date is not None
            else None
        ),
        "t_plus_part": (
            {"value": t_plus, "date": t_plus_date.isoformat()}
            if t_plus is not None and t_plus_date is not None
            else None
        ),
    }


def normalize_maturity(text: str) -> Dict[str, Optional[str]]:
    """Parse maturity text without fabricating an unknown day."""
    normalized = _normalize_text(text)
    for pattern in _MATURITY_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        month_name = match.group("month").casefold()
        month = _MONTH_ALIASES.get(month_name)
        if month is None:
            month = _MONTH_ALIASES.get(month_name[:3])
        if month is None:
            continue
        year_text = match.group("year")
        year = int(year_text)
        if len(year_text) == 2:
            year += 2000
        day_text = match.groupdict().get("day")
        if day_text is None:
            return {
                "maturity_text": match.group(0),
                "maturity_date": None,
                "maturity_ym": f"{year:04d}-{month:02d}",
            }
        day = int(day_text)
        try:
            maturity = date(year, month, day)
        except ValueError:
            return {
                "maturity_text": match.group(0),
                "maturity_date": None,
                "maturity_ym": f"{year:04d}-{month:02d}",
            }
        return {
            "maturity_text": match.group(0),
            "maturity_date": maturity.isoformat(),
            "maturity_ym": f"{year:04d}-{month:02d}",
        }
    return {"maturity_text": None, "maturity_date": None, "maturity_ym": None}


def is_valid_isin(candidate: str) -> bool:
    """Return whether a full ISIN candidate has a valid format and check digit."""
    candidate = _normalize_text(candidate).upper()
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", candidate, re.I):
        return False
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


def detect_partial_isins(text: str) -> List[str]:
    """Return short numeric fragments explicitly following an ISIN label."""
    normalized = _normalize_text(text)
    return list(dict.fromkeys(
        match.group("fragment")
        for match in _PARTIAL_ISIN_RE.finditer(normalized)
    ))


def extract_isins(text: str) -> Dict[str, List[Any]]:
    """Extract full ISIN candidates and short numeric fragments near ISIN labels."""
    normalized = _normalize_text(text)
    candidates = list(dict.fromkeys(match.group(0).upper() for match in _ISIN_RE.finditer(normalized)))
    valid = [candidate for candidate in candidates if is_valid_isin(candidate)]
    partial = detect_partial_isins(normalized)
    return {
        "valid": valid,
        "invalid": [candidate for candidate in candidates if candidate not in valid],
        "partial": partial,
    }
