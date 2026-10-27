"""Build evidence-grounded Phase 4 matching requests from extracted deals."""

import re
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bond_matching import (
    BondFieldEvidence,
    BondMatchRequest,
    BondMatchResult,
    match_bond,
)
from bond_reference import BondReferenceIndex
from isin_utils import (
    IsinCandidate,
    IsinEvidence,
    IsinExtractionResult,
    IsinStatus,
    extract_isin_from_message,
    normalize_isin_candidate,
)


_IDENTITY_FIELDS = (
    "isin_text",
    "partial_isin_text",
    "bond_identity_text",
    "issuer_text",
    "maturity_text",
    "coupon_rate",
)


def _source_references(
    deal: Mapping[str, Any],
    message_number_map: Mapping[str, str],
    field_name: str,
) -> List[Tuple[str, str]]:
    evidence = deal.get("evidence")
    citations = evidence.get(field_name) if isinstance(evidence, dict) else None
    if not isinstance(citations, list):
        return []
    result = []
    for citation in citations:
        message_id = message_number_map.get(citation) if isinstance(citation, str) else None
        if message_id:
            result.append((citation, message_id))
    return result


def _find_value_span(source_text: str, value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    exact_start = source_text.casefold().find(text.casefold())
    if exact_start >= 0:
        return exact_start, exact_start + len(text)

    expression = re.escape(text)
    expression = re.sub(r"\\\s+", r"\\s+", expression)
    match = re.search(expression, source_text, re.IGNORECASE)
    if match:
        return match.start(), match.end()

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = re.escape(str(value))
        match = re.search(rf"(?<![\d.]){number}(?:\s*%)?(?![\d.])", source_text)
        if match:
            return match.start(), match.end()
    return None


def _field_evidence(
    deal: Mapping[str, Any],
    messages_by_id: Mapping[str, Mapping[str, Any]],
    message_number_map: Mapping[str, str],
    field_name: str,
    value: Any,
) -> Optional[BondFieldEvidence]:
    if value is None:
        return None
    for citation, message_id in _source_references(
        deal, message_number_map, field_name
    ):
        message = messages_by_id.get(message_id)
        if message is None:
            continue
        source_text = message.get("message_text")
        if not isinstance(source_text, str):
            continue
        span = _find_value_span(source_text, value)
        if span is None:
            continue
        return BondFieldEvidence(
            field_name={
                "issuer_text": "issuer",
                "maturity_text": "maturity_date",
            }.get(field_name, field_name),
            message_id=message_id,
            source_message_text=source_text,
            start=span[0],
            end=span[1],
        )
    return None


def _isin_result(
    deal: Mapping[str, Any],
    parsed_messages: Sequence[Mapping[str, Any]],
    message_number_map: Mapping[str, str],
) -> Tuple[Optional[IsinExtractionResult], Optional[str]]:
    messages_by_id = {
        message.get("message_id"): message
        for message in parsed_messages
        if isinstance(message.get("message_id"), str)
    }
    references = []
    seen_messages = set()
    for field_name in _IDENTITY_FIELDS:
        for citation, message_id in _source_references(
            deal, message_number_map, field_name
        ):
            if message_id not in seen_messages and message_id in messages_by_id:
                seen_messages.add(message_id)
                references.append((citation, message_id))
    if not references:
        return None, None

    extracted = []
    for citation, message_id in references:
        source_text = messages_by_id[message_id].get("message_text", "")
        if not isinstance(source_text, str):
            continue
        result = extract_isin_from_message(source_text, message_id)
        for candidate in result.candidates:
            extracted.append((citation, message_id, candidate))
    if not extracted:
        return None, references[0][1]

    anchor_message_id = extracted[0][1]
    evidence_by_isin: Dict[str, List[IsinEvidence]] = {}
    candidate_by_isin: Dict[str, IsinCandidate] = {}
    for citation, message_id, candidate in extracted:
        candidate_by_isin.setdefault(candidate.normalized_identifier, candidate)
        for evidence in candidate.evidence:
            if message_id == anchor_message_id:
                linked_evidence = replace(
                    evidence,
                    relationship="direct_message",
                    linked_to_message_id=None,
                    link_evidence=None,
                )
            else:
                linked_evidence = replace(
                    evidence,
                    relationship="linked_message",
                    linked_to_message_id=anchor_message_id,
                    link_evidence=f"Deal identity evidence cited from {citation}.",
                )
            evidence_by_isin.setdefault(candidate.normalized_identifier, []).append(
                linked_evidence
            )

    candidates = tuple(
        replace(
            candidate_by_isin[normalized],
            evidence=tuple(evidence_by_isin[normalized]),
        )
        for normalized in evidence_by_isin
    )
    if len(candidates) > 1:
        status = IsinStatus.MULTIPLE_CANDIDATES
        selected = None
    else:
        selected = candidates[0]
        if not selected.format_valid:
            status = IsinStatus.INVALID_FORMAT
        elif not selected.checksum_valid:
            status = IsinStatus.CHECKSUM_FAILURE
        else:
            status = IsinStatus.VALID_ISIN
    source_evidence = tuple(
        evidence for candidate in candidates for evidence in candidate.evidence
    )
    result = IsinExtractionResult(
        status=status,
        original_identifier=(
            selected.original_identifier if selected is not None else None
        ),
        normalized_identifier=(
            selected.normalized_identifier if selected is not None else None
        ),
        candidates=candidates,
        source_evidence=source_evidence,
    )
    return result, anchor_message_id


def identify_extracted_deal(
    deal: Mapping[str, Any],
    parsed_messages: Sequence[Mapping[str, Any]],
    message_number_map: Mapping[str, str],
    reference_index: Optional[BondReferenceIndex],
    reference_error: Optional[str] = None,
) -> BondMatchResult:
    """Match only source values connected to this deal by its field citations."""
    messages_by_id = {
        message.get("message_id"): message
        for message in parsed_messages
        if isinstance(message.get("message_id"), str)
    }
    isin_result, identifier_message_id = _isin_result(
        deal, parsed_messages, message_number_map
    )
    issuer_evidence = _field_evidence(
        deal,
        messages_by_id,
        message_number_map,
        "issuer_text",
        deal.get("issuer_text"),
    )
    maturity_evidence = _field_evidence(
        deal,
        messages_by_id,
        message_number_map,
        "maturity_text",
        deal.get("maturity_text"),
    )
    coupon_evidence = _field_evidence(
        deal,
        messages_by_id,
        message_number_map,
        "coupon_rate",
        deal.get("coupon_rate"),
    )
    anchor_message_id = (
        identifier_message_id
        or (issuer_evidence.message_id if issuer_evidence else None)
        or (maturity_evidence.message_id if maturity_evidence else None)
        or (coupon_evidence.message_id if coupon_evidence else None)
        or ""
    )
    field_evidence = (
        issuer_evidence,
        maturity_evidence,
        coupon_evidence,
    )
    linked_evidence = []
    for evidence in field_evidence:
        if evidence is None:
            linked_evidence.append(None)
        elif evidence.message_id == anchor_message_id:
            linked_evidence.append(evidence)
        else:
            linked_evidence.append(
                replace(
                    evidence,
                    linked_to_message_id=anchor_message_id,
                    link_evidence=(
                        "The field is cited as evidence for the same extracted deal."
                    ),
                )
            )
    result = match_bond(
        BondMatchRequest(
            trade_message_id=anchor_message_id,
            isin_result=isin_result,
            raw_issuer_text=deal.get("issuer_text"),
            issuer_evidence=linked_evidence[0],
            maturity_date=deal.get("maturity_text"),
            maturity_evidence=linked_evidence[1],
            coupon_rate=deal.get("coupon_rate"),
            coupon_evidence=linked_evidence[2],
        ),
        reference_index,
    )
    if reference_error:
        result = replace(
            result,
            explanation=(
                "Bond reference data is unavailable. Check BOND_REFERENCE_CSV "
                "before using this identification result."
            ),
        )
    return result


def identification_summary(result: BondMatchResult) -> Dict[str, Any]:
    return {
        "status": result.status.value,
        "matching_method": result.matching_method,
        "explanation": result.explanation,
        "normalized_isin": result.normalized_isin,
        "raw_issuer_text": result.raw_issuer_text,
        "normalized_issuer_key": result.normalized_issuer_key,
        "matched_isin": (
            result.matched_reference.normalized_isin
            if result.matched_reference is not None
            else None
        ),
        "candidate_isins": [
            record.normalized_isin for record in result.candidate_references
        ],
    }
