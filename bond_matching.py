"""Evidence-grounded primary and secondary bond-reference matching."""

import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional, Tuple, Union

from bond_reference import (
    BondReferenceIndex,
    BondReferenceRecord,
    normalize_issuer_key,
    parse_reference_maturity_date,
)
from isin_utils import (
    IsinEvidence,
    IsinExtractionResult,
    IsinStatus,
    normalize_isin_candidate,
)
from normalize import is_valid_isin


class BondMatchStatus(str, Enum):
    PRIMARY_ISIN_MATCH = "primary_isin_match"
    VALID_ISIN_NOT_FOUND = "valid_isin_not_found"
    SECONDARY_ISSUER_MATCH = "secondary_issuer_match"
    NO_MATCH = "no_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INVALID_OR_UNSUPPORTED_IDENTIFIER = "invalid_or_unsupported_identifier"
    UNSUPPORTED_SECONDARY_EVIDENCE = "unsupported_secondary_evidence"
    REFERENCE_UNAVAILABLE = "reference_unavailable"


@dataclass(frozen=True)
class BondFieldEvidence:
    """Evidence locating one extracted value in a trade or explicitly linked message."""

    field_name: str
    message_id: str
    source_message_text: str
    start: int
    end: int
    linked_to_message_id: Optional[str] = None
    link_evidence: Optional[str] = None

    @property
    def source_value(self) -> str:
        return self.source_message_text[self.start:self.end]


@dataclass(frozen=True)
class BondMatchRequest:
    trade_message_id: str
    isin_result: Optional[IsinExtractionResult] = None
    raw_issuer_text: Optional[str] = None
    issuer_evidence: Optional[BondFieldEvidence] = None
    maturity_date: Optional[Union[date, str]] = None
    maturity_evidence: Optional[BondFieldEvidence] = None
    coupon_rate: Optional[Union[Decimal, int, float, str]] = None
    coupon_evidence: Optional[BondFieldEvidence] = None
    security_description: Optional[str] = None
    description_evidence: Optional[BondFieldEvidence] = None


@dataclass(frozen=True)
class BondMatchResult:
    status: BondMatchStatus
    identity_level: str
    matching_method: str
    attempted_methods: Tuple[str, ...]
    explanation: str
    matched_reference: Optional[BondReferenceRecord]
    candidate_references: Tuple[BondReferenceRecord, ...]
    isin_status: Optional[IsinStatus]
    normalized_isin: Optional[str]
    raw_issuer_text: Optional[str]
    normalized_issuer_key: Optional[str]
    source_evidence: Tuple[BondFieldEvidence, ...]
    isin_source_evidence: Tuple[IsinEvidence, ...]


@dataclass(frozen=True)
class _SecondaryResult:
    candidates: Tuple[BondReferenceRecord, ...]
    issuer_candidates: Tuple[BondReferenceRecord, ...]
    explanation: str


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def _parse_maturity(value: Union[date, str]) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return parse_reference_maturity_date(value)
    except ValueError:
        return None


def _parse_rate(value: Union[Decimal, int, float, str]) -> Optional[Decimal]:
    if isinstance(value, bool):
        return None
    try:
        normalized = str(value).strip().removesuffix("%")
        rate = Decimal(normalized)
        return rate if rate.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _evidence_is_supported(
    evidence: Optional[BondFieldEvidence],
    field_name: str,
    trade_message_id: str,
) -> bool:
    if evidence is None or evidence.field_name != field_name:
        return False
    if (
        not evidence.message_id
        or not isinstance(evidence.source_message_text, str)
        or evidence.start < 0
        or evidence.end <= evidence.start
        or evidence.end > len(evidence.source_message_text)
    ):
        return False
    if evidence.linked_to_message_id is None:
        return (
            evidence.message_id == trade_message_id
            and evidence.link_evidence is None
        )
    return (
        evidence.linked_to_message_id == trade_message_id
        and isinstance(evidence.link_evidence, str)
        and bool(evidence.link_evidence.strip())
    )


def _isin_evidence_is_supported(
    result: IsinExtractionResult,
    trade_message_id: str,
) -> bool:
    if not result.candidates or not result.source_evidence:
        return False
    candidate_evidence = {
        evidence
        for candidate in result.candidates
        for evidence in candidate.evidence
    }
    if any(evidence not in candidate_evidence for evidence in result.source_evidence):
        return False
    return all(
        evidence.start >= 0
        and evidence.end > evidence.start
        and evidence.end <= len(evidence.source_message_text)
        and evidence.source_message_text[evidence.start:evidence.end]
        == evidence.matched_text
        and normalize_isin_candidate(evidence.matched_text)
        == normalize_isin_candidate(
            next(
                candidate.normalized_identifier
                for candidate in result.candidates
                if evidence in candidate.evidence
            )
        )
        and (
            (
                evidence.relationship == "direct_message"
                and evidence.message_id == trade_message_id
                and evidence.linked_to_message_id is None
            )
            or (
                evidence.relationship == "linked_message"
                and evidence.linked_to_message_id == trade_message_id
                and bool(evidence.link_evidence and evidence.link_evidence.strip())
            )
        )
        for evidence in result.source_evidence
    )


def _request_field_evidence(
    request: BondMatchRequest,
) -> Tuple[Tuple[BondFieldEvidence, ...], Optional[str]]:
    checks = (
        ("issuer", request.raw_issuer_text, request.issuer_evidence, normalize_issuer_key),
        (
            "maturity_date",
            request.maturity_date,
            request.maturity_evidence,
            _parse_maturity,
        ),
        (
            "coupon_rate",
            request.coupon_rate,
            request.coupon_evidence,
            _parse_rate,
        ),
        (
            "security_description",
            request.security_description,
            request.description_evidence,
            _normalized_text,
        ),
    )
    accepted = []
    for field_name, raw_value, evidence, normalize in checks:
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            if evidence is not None:
                return (), f"Evidence was supplied without a {field_name} value."
            continue
        expected_field = "issuer" if field_name == "issuer" else field_name
        if not _evidence_is_supported(
            evidence,
            expected_field,
            request.trade_message_id,
        ):
            return (), f"The {field_name} value is not supported by linked source evidence."
        if normalize(evidence.source_value) != normalize(str(raw_value)):
            return (), f"The {field_name} value does not match its source evidence."
        if field_name == "maturity_date" and _parse_maturity(raw_value) is None:
            return (), "The source maturity value could not be parsed."
        if field_name == "coupon_rate" and _parse_rate(raw_value) is None:
            return (), "The source coupon/rate value could not be parsed."
        accepted.append(evidence)
    return tuple(accepted), None


def _secondary_candidates(request: BondMatchRequest, index: BondReferenceIndex) -> _SecondaryResult:
    if not request.raw_issuer_text or not request.raw_issuer_text.strip():
        return _SecondaryResult((), (), "No issuer text was supplied.")

    issuer_candidates = index.candidates_by_issuer(request.raw_issuer_text)
    if not issuer_candidates:
        return _SecondaryResult(
            (), (), "No exact normalized issuer-key match exists in the reference index."
        )

    comparators = []
    if request.maturity_date is not None:
        comparators.append(
            (
                lambda record: record.maturity_date,
                _parse_maturity(request.maturity_date),
            )
        )
    if request.coupon_rate is not None:
        comparators.append(
            (
                lambda record: _parse_rate(
                    record.source_values.get("INTEREST_RATE", "")
                ),
                _parse_rate(request.coupon_rate),
            )
        )
    if request.security_description and request.security_description.strip():
        comparators.append(
            (
                lambda record: _normalized_text(
                    record.source_values.get("SECURITY_DESCRIPTION", "")
                ),
                _normalized_text(request.security_description),
            )
        )

    if not comparators:
        return _SecondaryResult(
            (), issuer_candidates,
            "Exact issuer candidates exist, but no corroborating maturity, "
            "coupon/rate, or security description was supplied.",
        )

    matches = []
    for record in issuer_candidates:
        matched_fields = 0
        contradicted = False
        for get_reference_value, supplied_value in comparators:
            reference_value = get_reference_value(record)
            if reference_value is None or supplied_value is None:
                continue
            if reference_value != supplied_value:
                contradicted = True
                break
            matched_fields += 1
        if not contradicted and matched_fields:
            matches.append(record)

    if not matches:
        explanation = (
            "Every exact issuer candidate was contradicted by supplied "
            "corroborating details, or lacked a comparable reference value."
        )
    else:
        explanation = (
            "Candidates were filtered by exact normalized issuer and all "
            "comparable supplied corroborating fields."
        )
    return _SecondaryResult(
        tuple(matches),
        issuer_candidates,
        explanation,
    )


def _make_result(
    request: BondMatchRequest,
    status: BondMatchStatus,
    matching_method: str,
    attempted_methods: Tuple[str, ...],
    explanation: str,
    matched_reference: Optional[BondReferenceRecord] = None,
    candidate_references: Tuple[BondReferenceRecord, ...] = (),
    isin_status: Optional[IsinStatus] = None,
    normalized_isin: Optional[str] = None,
    source_evidence: Tuple[BondFieldEvidence, ...] = (),
) -> BondMatchResult:
    identity_levels = {
        BondMatchStatus.PRIMARY_ISIN_MATCH: "ISIN-confirmed",
        BondMatchStatus.SECONDARY_ISSUER_MATCH:
            "uniquely resolved by secondary details",
        BondMatchStatus.AMBIGUOUS_MATCH: "ambiguous",
    }
    return BondMatchResult(
        status=status,
        identity_level=identity_levels.get(status, "unknown"),
        matching_method=matching_method,
        attempted_methods=attempted_methods,
        explanation=explanation,
        matched_reference=matched_reference,
        candidate_references=candidate_references,
        isin_status=isin_status,
        normalized_isin=normalized_isin,
        raw_issuer_text=request.raw_issuer_text,
        normalized_issuer_key=(
            normalize_issuer_key(request.raw_issuer_text)
            if request.raw_issuer_text
            else None
        ),
        source_evidence=source_evidence,
        isin_source_evidence=(
            request.isin_result.source_evidence
            if request.isin_result is not None
            else ()
        ),
    )


def match_bond(
    request: BondMatchRequest,
    reference_index: Optional[BondReferenceIndex],
) -> BondMatchResult:
    """Match source-supported evidence without selecting among competing records."""
    attempted = []
    isin_result = request.isin_result
    isin_status = isin_result.status if isin_result is not None else None
    normalized_isin = (
        isin_result.normalized_identifier if isin_result is not None else None
    )

    if reference_index is None:
        return _make_result(
            request,
            BondMatchStatus.REFERENCE_UNAVAILABLE,
            "none",
            (),
            "No bond reference index was supplied.",
            isin_status=isin_status,
            normalized_isin=normalized_isin,
        )

    evidence, evidence_error = _request_field_evidence(request)
    primary_record = None
    primary_candidates: Tuple[BondReferenceRecord, ...] = ()
    if isin_result is not None:
        if isin_status in {
            IsinStatus.INVALID_FORMAT,
            IsinStatus.CHECKSUM_FAILURE,
            IsinStatus.UNSUPPORTED_BY_SOURCE,
        }:
            return _make_result(
                request,
                BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
                "none",
                ("primary_isin",),
                f"Phase 3 identifier status is {isin_status.value}; exact ISIN lookup was not attempted.",
                isin_status=isin_status,
                normalized_isin=normalized_isin,
                source_evidence=evidence,
            )
        if isin_status != IsinStatus.NO_IDENTIFIER and not _isin_evidence_is_supported(
            isin_result,
            request.trade_message_id,
        ):
            return _make_result(
                request,
                BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
                "none",
                ("primary_isin",),
                "ISIN candidate evidence does not belong to this trade message "
                "or an explicitly linked message.",
                isin_status=isin_status,
                normalized_isin=normalized_isin,
                source_evidence=evidence,
            )
        if isin_status == IsinStatus.MULTIPLE_CANDIDATES:
            attempted.append("primary_isin")
            candidates = tuple(
                record
                for candidate in isin_result.candidates
                if candidate.format_valid and candidate.checksum_valid is True
                for record in reference_index.lookup_isin_candidates(
                    candidate.normalized_identifier
                )
            )
            return _make_result(
                request,
                BondMatchStatus.AMBIGUOUS_MATCH,
                "primary_isin",
                tuple(attempted),
                "The source contains multiple ISIN candidates; none was selected.",
                candidate_references=candidates,
                isin_status=isin_status,
                source_evidence=evidence,
            )
        if isin_status == IsinStatus.VALID_ISIN:
            if (
                len(isin_result.candidates) != 1
                or not isin_result.candidates[0].format_valid
                or isin_result.candidates[0].checksum_valid is not True
                or not normalized_isin
                or normalize_isin_candidate(
                    isin_result.candidates[0].normalized_identifier
                ) != normalize_isin_candidate(normalized_isin)
                or not is_valid_isin(normalized_isin)
            ):
                return _make_result(
                    request,
                    BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
                    "none",
                    ("primary_isin",),
                    "The Phase 3 valid status is missing a single checksum-valid "
                    "candidate supported by this trade's source evidence.",
                    isin_status=isin_status,
                    normalized_isin=normalized_isin,
                    source_evidence=evidence,
                )
            attempted.append("primary_isin")
            primary_candidates = reference_index.lookup_isin_candidates(normalized_isin)
            primary_record = reference_index.lookup_isin(normalized_isin)
            if len(primary_candidates) > 1 and primary_record is None:
                return _make_result(
                    request,
                    BondMatchStatus.AMBIGUOUS_MATCH,
                    "primary_isin",
                    tuple(attempted),
                    "The exact ISIN has conflicting reference records.",
                    candidate_references=primary_candidates,
                    isin_status=isin_status,
                    normalized_isin=normalized_isin,
                    source_evidence=evidence,
                )
        elif isin_status != IsinStatus.NO_IDENTIFIER:
            return _make_result(
                request,
                BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
                "none",
                ("primary_isin",),
                f"Unsupported Phase 3 identifier status: {isin_status.value}.",
                isin_status=isin_status,
                normalized_isin=normalized_isin,
                source_evidence=evidence,
            )

    if evidence_error:
        if primary_record is not None:
            return _make_result(
                request,
                BondMatchStatus.PRIMARY_ISIN_MATCH,
                "primary_isin",
                tuple(attempted),
                "The source-supported ISIN exactly matched the reference index. "
                f"Secondary details were ignored: {evidence_error}",
                matched_reference=primary_record,
                candidate_references=(primary_record,),
                isin_status=isin_status,
                normalized_isin=normalized_isin,
                source_evidence=evidence,
            )
        if isin_status == IsinStatus.VALID_ISIN:
            return _make_result(
                request,
                BondMatchStatus.VALID_ISIN_NOT_FOUND,
                "primary_isin",
                tuple(attempted),
                "The valid, source-supported ISIN is absent from the reference index. "
                f"Secondary details were ignored: {evidence_error}",
                isin_status=isin_status,
                normalized_isin=normalized_isin,
                source_evidence=evidence,
            )
        return _make_result(
            request,
            BondMatchStatus.UNSUPPORTED_SECONDARY_EVIDENCE,
            "none",
            tuple(attempted),
            evidence_error,
            isin_status=isin_status,
            normalized_isin=normalized_isin,
        )

    issuer_supplied = bool(request.raw_issuer_text and request.raw_issuer_text.strip())
    secondary = _SecondaryResult((), (), "Secondary matching was not attempted.")
    if issuer_supplied:
        attempted.append("secondary_issuer")
        secondary = _secondary_candidates(request, reference_index)

    if primary_record is not None:
        if len(secondary.candidates) == 1:
            secondary_record = secondary.candidates[0]
            if secondary_record.normalized_isin != primary_record.normalized_isin:
                return _make_result(
                    request,
                    BondMatchStatus.CONFLICTING_EVIDENCE,
                    "conflict",
                    tuple(attempted),
                    "The source-supported ISIN and exact-issuer corroborating "
                    "details identify different reference securities; neither was selected.",
                    candidate_references=(primary_record, secondary_record),
                    isin_status=isin_status,
                    normalized_isin=normalized_isin,
                    source_evidence=evidence,
                )
        return _make_result(
            request,
            BondMatchStatus.PRIMARY_ISIN_MATCH,
            "primary_isin",
            tuple(attempted),
            (
                "The source-supported, checksum-valid ISIN exactly matched "
                "the reference index."
                if not issuer_supplied or not secondary.candidates
                else "The source-supported ISIN matched the reference; secondary "
                     "details did not identify a conflicting security."
            ),
            matched_reference=primary_record,
            candidate_references=(primary_record,),
            isin_status=isin_status,
            normalized_isin=normalized_isin,
            source_evidence=evidence,
        )

    if len(secondary.candidates) > 1:
        return _make_result(
            request,
            BondMatchStatus.AMBIGUOUS_MATCH,
            "secondary_issuer",
            tuple(attempted),
            f"{len(secondary.candidates)} reference securities match the exact "
            f"issuer and supported corroborating details.",
            candidate_references=secondary.candidates,
            isin_status=isin_status,
            normalized_isin=normalized_isin,
            source_evidence=evidence,
        )
    if len(secondary.candidates) == 1:
        secondary_record = secondary.candidates[0]
        if normalized_isin and (
            isin_status == IsinStatus.VALID_ISIN
            and secondary_record.normalized_isin != normalized_isin
        ):
            return _make_result(
                request,
                BondMatchStatus.CONFLICTING_EVIDENCE,
                "conflict",
                tuple(attempted),
                "The source-supported valid ISIN was absent from the reference "
                "index, while issuer details matched a different reference ISIN.",
                candidate_references=(secondary_record,),
                isin_status=isin_status,
                normalized_isin=normalized_isin,
                source_evidence=evidence,
            )
        return _make_result(
            request,
            BondMatchStatus.SECONDARY_ISSUER_MATCH,
            "secondary_issuer",
            tuple(attempted),
            secondary.explanation,
            matched_reference=secondary_record,
            candidate_references=(secondary_record,),
            isin_status=isin_status,
            normalized_isin=normalized_isin,
            source_evidence=evidence,
        )

    if isin_status == IsinStatus.VALID_ISIN:
        return _make_result(
            request,
            BondMatchStatus.VALID_ISIN_NOT_FOUND,
            "primary_isin",
            tuple(attempted),
            "The valid, source-supported ISIN is absent from the reference index. "
            + secondary.explanation,
            isin_status=isin_status,
            normalized_isin=normalized_isin,
            source_evidence=evidence,
        )
    return _make_result(
        request,
        BondMatchStatus.NO_MATCH,
        "secondary_issuer" if issuer_supplied else "none",
        tuple(attempted),
        secondary.explanation if issuer_supplied else "No supported identifier or issuer evidence was supplied.",
        candidate_references=secondary.issuer_candidates,
        isin_status=isin_status,
        normalized_isin=normalized_isin,
        source_evidence=evidence,
    )
