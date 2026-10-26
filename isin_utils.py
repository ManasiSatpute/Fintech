"""Source-grounded extraction and validation of ISIN candidates."""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

from normalize import is_valid_isin

_ISIN_SHAPE_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2}(?:[\s-]*[A-Za-z0-9]){9}[\s-]*\d"
    r"(?![A-Za-z0-9])"
)
_UNLABELED_NEAR_ISIN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2}[A-Za-z0-9]{4,14}(?![A-Za-z0-9])"
)
_LABELED_CANDIDATE_RE = re.compile(
    r"\bISIN\b\s*[:#=]?\s*"
    r"(?P<candidate>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)",
    re.IGNORECASE,
)
_NORMALIZED_ISIN_FORMAT_RE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}\d")


class IsinStatus(str, Enum):
    NO_IDENTIFIER = "no_identifier"
    VALID_ISIN = "valid_isin"
    INVALID_FORMAT = "invalid_format"
    CHECKSUM_FAILURE = "checksum_failure"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    UNSUPPORTED_BY_SOURCE = "unsupported_by_source"


@dataclass(frozen=True)
class IsinEvidence:
    """The exact match and its location in one original source message."""

    message_id: Optional[str]
    source_message_text: str
    start: int
    end: int
    matched_text: str
    relationship: str
    linked_to_message_id: Optional[str] = None
    link_evidence: Optional[str] = None


@dataclass(frozen=True)
class IsinCandidate:
    original_identifier: str
    normalized_identifier: str
    format_valid: bool
    checksum_valid: Optional[bool]
    evidence: Tuple[IsinEvidence, ...]


@dataclass(frozen=True)
class IsinExtractionResult:
    status: IsinStatus
    original_identifier: Optional[str]
    normalized_identifier: Optional[str]
    candidates: Tuple[IsinCandidate, ...]
    source_evidence: Tuple[IsinEvidence, ...]


@dataclass(frozen=True)
class LinkedMessageEvidence:
    """A source message explicitly linked to the original trade message."""

    message_id: str
    message_text: str
    linked_to_message_id: str
    link_evidence: str


def normalize_isin_candidate(candidate: str) -> str:
    """Remove permitted whitespace/hyphen separators and uppercase the value."""
    normalized = unicodedata.normalize("NFKC", candidate)
    return "".join(character for character in normalized if not character.isspace() and character != "-").upper()


def _candidate_matches(
    message_text: str,
) -> Tuple[Tuple[str, int, int], ...]:
    matches: Dict[Tuple[int, int], str] = {}
    for pattern in (_ISIN_SHAPE_RE, _UNLABELED_NEAR_ISIN_RE):
        for match in pattern.finditer(message_text):
            original = match.group(0)
            normalized = normalize_isin_candidate(original)
            if any(character.isdigit() for character in normalized):
                matches[(match.start(), match.end())] = original

    for match in _LABELED_CANDIDATE_RE.finditer(message_text):
        original = match.group("candidate")
        normalized = normalize_isin_candidate(original)
        if normalized and any(character.isdigit() for character in normalized):
            start, end = match.span("candidate")
            matches[(start, end)] = original

    return tuple(
        (original, start, end)
        for (start, end), original in sorted(matches.items())
    )


def _candidate_result(
    message_text: str,
    message_id: Optional[str],
    relationship: str,
    linked_to_message_id: Optional[str] = None,
    link_evidence: Optional[str] = None,
) -> Tuple[IsinCandidate, ...]:
    candidates: Dict[str, list[IsinEvidence]] = {}
    originals: Dict[str, str] = {}
    for original, start, end in _candidate_matches(message_text):
        normalized = normalize_isin_candidate(original)
        originals.setdefault(normalized, original)
        candidates.setdefault(normalized, []).append(
            IsinEvidence(
                message_id=message_id,
                source_message_text=message_text,
                start=start,
                end=end,
                matched_text=original,
                relationship=relationship,
                linked_to_message_id=linked_to_message_id,
                link_evidence=link_evidence,
            )
        )

    results = []
    for normalized, evidences in candidates.items():
        format_valid = bool(_NORMALIZED_ISIN_FORMAT_RE.fullmatch(normalized))
        checksum_valid = is_valid_isin(normalized) if format_valid else None
        results.append(
            IsinCandidate(
                original_identifier=originals[normalized],
                normalized_identifier=normalized,
                format_valid=format_valid,
                checksum_valid=checksum_valid,
                evidence=tuple(evidences),
            )
        )
    return tuple(results)


def _result_from_candidates(
    candidates: Iterable[IsinCandidate],
) -> IsinExtractionResult:
    candidate_tuple = tuple(candidates)
    evidence = tuple(
        occurrence
        for candidate in candidate_tuple
        for occurrence in candidate.evidence
    )
    if not candidate_tuple:
        return IsinExtractionResult(
            status=IsinStatus.NO_IDENTIFIER,
            original_identifier=None,
            normalized_identifier=None,
            candidates=(),
            source_evidence=(),
        )
    if len(candidate_tuple) > 1:
        status = IsinStatus.MULTIPLE_CANDIDATES
        selected = None
    else:
        selected = candidate_tuple[0]
        if not selected.format_valid:
            status = IsinStatus.INVALID_FORMAT
        elif not selected.checksum_valid:
            status = IsinStatus.CHECKSUM_FAILURE
        else:
            status = IsinStatus.VALID_ISIN
    return IsinExtractionResult(
        status=status,
        original_identifier=(
            selected.original_identifier if selected is not None else None
        ),
        normalized_identifier=(
            selected.normalized_identifier if selected is not None else None
        ),
        candidates=candidate_tuple,
        source_evidence=evidence,
    )


def extract_isin_from_message(
    message_text: str,
    message_id: Optional[str] = None,
) -> IsinExtractionResult:
    """Extract candidates from exactly one original WhatsApp message."""
    if not isinstance(message_text, str):
        raise TypeError("message_text must be a string")
    candidates = _candidate_result(
        message_text=message_text,
        message_id=message_id,
        relationship="direct_message",
    )
    return _result_from_candidates(candidates)


def validate_llm_isin(
    llm_candidate: str,
    original_message_text: str,
    original_message_id: Optional[str] = None,
    linked_messages: Iterable[LinkedMessageEvidence] = (),
) -> IsinExtractionResult:
    """Accept an LLM candidate only when original or explicitly linked text supports it.

    Linked messages must carry a target message id and non-empty link evidence.
    The caller is responsible for supplying only conversation messages linked to
    the original trade message, not an unfiltered chat window.
    """
    if not isinstance(llm_candidate, str):
        raise TypeError("llm_candidate must be a string")
    if not isinstance(original_message_text, str):
        raise TypeError("original_message_text must be a string")
    candidates = list(
        _candidate_result(
            original_message_text,
            original_message_id,
            "direct_message",
        )
    )
    for linked in tuple(linked_messages):
        if not isinstance(linked, LinkedMessageEvidence):
            raise TypeError(
                "linked_messages entries must be LinkedMessageEvidence instances"
            )
        if (
            not original_message_id
            or linked.linked_to_message_id != original_message_id
            or not isinstance(linked.link_evidence, str)
            or not linked.link_evidence.strip()
            or not isinstance(linked.message_text, str)
            or not linked.message_id
        ):
            raise ValueError(
                "Linked message evidence must reference the original message "
                "and include its explicit link evidence."
            )
        candidates.extend(
            _candidate_result(
                linked.message_text,
                linked.message_id,
                "linked_message",
                linked.linked_to_message_id,
                linked.link_evidence,
            )
        )

    by_normalized: Dict[str, IsinCandidate] = {}
    for candidate in candidates:
        previous = by_normalized.get(candidate.normalized_identifier)
        if previous is None:
            by_normalized[candidate.normalized_identifier] = candidate
        else:
            by_normalized[candidate.normalized_identifier] = IsinCandidate(
                original_identifier=previous.original_identifier,
                normalized_identifier=previous.normalized_identifier,
                format_valid=previous.format_valid,
                checksum_valid=previous.checksum_valid,
                evidence=previous.evidence + candidate.evidence,
            )

    normalized_llm_candidate = normalize_isin_candidate(llm_candidate)
    supported = normalized_llm_candidate in by_normalized
    result = _result_from_candidates(by_normalized.values())
    if not supported:
        return IsinExtractionResult(
            status=IsinStatus.UNSUPPORTED_BY_SOURCE,
            original_identifier=llm_candidate,
            normalized_identifier=normalized_llm_candidate,
            candidates=result.candidates,
            source_evidence=result.source_evidence,
        )
    return result
