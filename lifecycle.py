"""
Transaction Lifecycle Management Module for Bond Trade Extraction.

Defines:
1. Business lifecycle statuses and event types.
2. Pydantic models for lifecycle events and state transitions.
3. Deterministic state transition engine with transition validation,
   terminal state protection, and late-arriving message protection.
4. Semantic / linguistic lifecycle event classification supporting English,
   Hinglish, and bond-market abbreviations.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. LIFECYCLE STATUSES AND EVENT TYPES
# --------------------------------------------------------------------------

class LifecycleStatus(str, Enum):
    """Business lifecycle status of a bond transaction."""

    INITIATED = "INITIATED"
    NEGOTIATING = "NEGOTIATING"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    CONCLUDED = "CONCLUDED"
    CANCELLED = "CANCELLED"
    ON_HOLD = "ON_HOLD"
    UNCERTAIN = "UNCERTAIN"
    DISCARDED = "DISCARDED"


class LifecycleEventType(str, Enum):
    """Types of lifecycle events extracted from messages."""

    INQUIRY = "INQUIRY"
    NEGOTIATION = "NEGOTIATION"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    DEAL_CONFIRMED = "DEAL_CONFIRMED"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    TRADE_CANCELLED = "TRADE_CANCELLED"
    TRADE_REJECTED = "TRADE_REJECTED"
    TRADE_ON_HOLD = "TRADE_ON_HOLD"
    TRADE_CORRECTED = "TRADE_CORRECTED"
    NO_LIFECYCLE_CHANGE = "NO_LIFECYCLE_CHANGE"
    AMBIGUOUS = "AMBIGUOUS"


# Terminal states cannot be altered by normal lifecycle events.
TERMINAL_STATUSES: Set[LifecycleStatus] = {
    LifecycleStatus.CONCLUDED,
    LifecycleStatus.CANCELLED,
}

# Active states that can transition through normal progression.
ACTIVE_STATUSES: Set[LifecycleStatus] = {
    LifecycleStatus.INITIATED,
    LifecycleStatus.NEGOTIATING,
    LifecycleStatus.PENDING_CONFIRMATION,
    LifecycleStatus.ON_HOLD,
}


# --------------------------------------------------------------------------
# 2. PYDANTIC MODELS
# --------------------------------------------------------------------------

class LifecycleEvent(BaseModel):
    """Structured representation of a lifecycle event extracted from a message."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: LifecycleEventType
    message_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_key: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    event_timestamp: Optional[str] = None
    transaction_reference: Optional[str] = None
    isin: Optional[str] = None
    issuer: Optional[str] = None
    action: Optional[str] = None  # BUY, SELL, TWO_WAY
    quantity: Optional[float] = None
    ytm: Optional[float] = None
    price: Optional[float] = None
    evidence_text: str = ""
    confidence_score: float = 1.0
    requires_review: bool = False
    is_correction: bool = False
    correction_reason: Optional[str] = None
    target_transaction_id: Optional[str] = None


class TransitionResult(BaseModel):
    """Result of attempting a state transition."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    previous_status: LifecycleStatus
    new_status: LifecycleStatus
    event_type: LifecycleEventType
    reason: str
    requires_review: bool
    is_terminal: bool


# --------------------------------------------------------------------------
# 3. DETERMINISTIC STATE TRANSITION ENGINE
# --------------------------------------------------------------------------

# Final transition matrix: maps (current_status, event_type) -> (target_status, requires_explicit_evidence)
# Note: TRADE_CORRECTED is handled separately via explicit correction validation.
_TRANSITION_RULES: Dict[Tuple[LifecycleStatus, LifecycleEventType], LifecycleStatus] = {
    # From INITIATED
    (LifecycleStatus.INITIATED, LifecycleEventType.INQUIRY): LifecycleStatus.INITIATED,
    (LifecycleStatus.INITIATED, LifecycleEventType.NEGOTIATION): LifecycleStatus.NEGOTIATING,
    (LifecycleStatus.INITIATED, LifecycleEventType.AWAITING_CONFIRMATION): LifecycleStatus.PENDING_CONFIRMATION,
    (LifecycleStatus.INITIATED, LifecycleEventType.DEAL_CONFIRMED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.INITIATED, LifecycleEventType.TRADE_EXECUTED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.INITIATED, LifecycleEventType.TRADE_CANCELLED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.INITIATED, LifecycleEventType.TRADE_REJECTED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.INITIATED, LifecycleEventType.TRADE_ON_HOLD): LifecycleStatus.ON_HOLD,
    (LifecycleStatus.INITIATED, LifecycleEventType.NO_LIFECYCLE_CHANGE): LifecycleStatus.INITIATED,

    # From NEGOTIATING
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.INQUIRY): LifecycleStatus.NEGOTIATING,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.NEGOTIATION): LifecycleStatus.NEGOTIATING,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.AWAITING_CONFIRMATION): LifecycleStatus.PENDING_CONFIRMATION,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.DEAL_CONFIRMED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.TRADE_EXECUTED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.TRADE_CANCELLED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.TRADE_REJECTED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.TRADE_ON_HOLD): LifecycleStatus.ON_HOLD,
    (LifecycleStatus.NEGOTIATING, LifecycleEventType.NO_LIFECYCLE_CHANGE): LifecycleStatus.NEGOTIATING,

    # From PENDING_CONFIRMATION
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.INQUIRY): LifecycleStatus.PENDING_CONFIRMATION,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.NEGOTIATION): LifecycleStatus.NEGOTIATING,  # reopened negotiation
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.AWAITING_CONFIRMATION): LifecycleStatus.PENDING_CONFIRMATION,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.DEAL_CONFIRMED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.TRADE_EXECUTED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.TRADE_CANCELLED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.TRADE_REJECTED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.TRADE_ON_HOLD): LifecycleStatus.ON_HOLD,
    (LifecycleStatus.PENDING_CONFIRMATION, LifecycleEventType.NO_LIFECYCLE_CHANGE): LifecycleStatus.PENDING_CONFIRMATION,

    # From ON_HOLD
    (LifecycleStatus.ON_HOLD, LifecycleEventType.INQUIRY): LifecycleStatus.INITIATED,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.NEGOTIATION): LifecycleStatus.NEGOTIATING,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.AWAITING_CONFIRMATION): LifecycleStatus.PENDING_CONFIRMATION,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.DEAL_CONFIRMED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.TRADE_EXECUTED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.TRADE_CANCELLED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.TRADE_REJECTED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.TRADE_ON_HOLD): LifecycleStatus.ON_HOLD,
    (LifecycleStatus.ON_HOLD, LifecycleEventType.NO_LIFECYCLE_CHANGE): LifecycleStatus.ON_HOLD,

    # From UNCERTAIN
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.INQUIRY): LifecycleStatus.INITIATED,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.NEGOTIATION): LifecycleStatus.NEGOTIATING,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.AWAITING_CONFIRMATION): LifecycleStatus.PENDING_CONFIRMATION,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.DEAL_CONFIRMED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.TRADE_EXECUTED): LifecycleStatus.CONCLUDED,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.TRADE_CANCELLED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.TRADE_REJECTED): LifecycleStatus.CANCELLED,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.TRADE_ON_HOLD): LifecycleStatus.ON_HOLD,
    (LifecycleStatus.UNCERTAIN, LifecycleEventType.NO_LIFECYCLE_CHANGE): LifecycleStatus.UNCERTAIN,

    # From DISCARDED
    (LifecycleStatus.DISCARDED, LifecycleEventType.NO_LIFECYCLE_CHANGE): LifecycleStatus.DISCARDED,
}

# Relative progression rank used to detect late-arriving regression
_STATUS_PROGRESSION_RANK: Dict[LifecycleStatus, int] = {
    LifecycleStatus.DISCARDED: 0,
    LifecycleStatus.UNCERTAIN: 1,
    LifecycleStatus.INITIATED: 2,
    LifecycleStatus.ON_HOLD: 3,
    LifecycleStatus.NEGOTIATING: 4,
    LifecycleStatus.PENDING_CONFIRMATION: 5,
    LifecycleStatus.CONCLUDED: 6,
    LifecycleStatus.CANCELLED: 6,
}


def _parse_iso_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


class LifecycleEngine:
    """
    Deterministic state transition engine for bond trade lifecycle management.
    Independent of LLM and database; fully unit-testable.
    """

    @staticmethod
    def get_allowed_transitions() -> Dict[Tuple[LifecycleStatus, LifecycleEventType], LifecycleStatus]:
        """Return the transition matrix."""
        return dict(_TRANSITION_RULES)

    @classmethod
    def validate_transition(
        cls,
        current_status: LifecycleStatus | str,
        event: LifecycleEvent,
        last_update_timestamp: Optional[str] = None,
    ) -> TransitionResult:
        """
        Validate whether an event can be applied to a transaction in current_status.

        Rules:
        1. Reject or flag unsupported transitions.
        2. Do not silently overwrite terminal states (CONCLUDED, CANCELLED).
        3. A correction to a terminal trade requires explicit correction evidence and rule.
        4. Late-arriving messages must not automatically regress a transaction.
        5. Ambiguous events leave the status unchanged and set requires_review=True.
        6. Return detailed justification.
        """
        try:
            current = (
                current_status
                if isinstance(current_status, LifecycleStatus)
                else LifecycleStatus(str(current_status).upper())
            )
        except ValueError:
            return TransitionResult(
                allowed=False,
                previous_status=LifecycleStatus.UNCERTAIN,
                new_status=LifecycleStatus.UNCERTAIN,
                event_type=event.event_type,
                reason=f"Invalid current lifecycle status: '{current_status}'.",
                requires_review=True,
                is_terminal=False,
            )

        # Ambiguous events never change status; always require review
        if event.event_type == LifecycleEventType.AMBIGUOUS:
            return TransitionResult(
                allowed=True,
                previous_status=current,
                new_status=current,
                event_type=event.event_type,
                reason="Event is ambiguous; current status preserved. Requires human review.",
                requires_review=True,
                is_terminal=current in TERMINAL_STATUSES,
            )

        # Low confidence requires review without mutating state
        if event.confidence_score < config.REVIEW_CONFIDENCE_THRESHOLD:
            return TransitionResult(
                allowed=True,
                previous_status=current,
                new_status=current,
                event_type=event.event_type,
                reason=(
                    f"Event confidence ({event.confidence_score:.2f}) is below threshold "
                    f"({config.REVIEW_CONFIDENCE_THRESHOLD:.2f}); current status preserved. "
                    "Requires human review."
                ),
                requires_review=True,
                is_terminal=current in TERMINAL_STATUSES,
            )

        # Explicit correction handling
        if event.event_type == LifecycleEventType.TRADE_CORRECTED or event.is_correction:
            if not event.correction_reason:
                return TransitionResult(
                    allowed=False,
                    previous_status=current,
                    new_status=current,
                    event_type=event.event_type,
                    reason="Trade correction requires an explicit correction reason.",
                    requires_review=True,
                    is_terminal=current in TERMINAL_STATUSES,
                )
            # A correction can transition to a specified valid target status or to UNCERTAIN for review
            target = LifecycleStatus.UNCERTAIN
            if event.evidence_text and "CANCEL" in event.evidence_text.upper():
                target = LifecycleStatus.CANCELLED
            elif event.evidence_text and ("CONFIRM" in event.evidence_text.upper() or "EXECUTE" in event.evidence_text.upper()):
                target = LifecycleStatus.CONCLUDED
            elif current == LifecycleStatus.CONCLUDED or current == LifecycleStatus.CANCELLED:
                target = LifecycleStatus.NEGOTIATING  # reopen for renegotiation upon correction

            return TransitionResult(
                allowed=True,
                previous_status=current,
                new_status=target,
                event_type=LifecycleEventType.TRADE_CORRECTED,
                reason=f"Explicit correction applied: {event.correction_reason}",
                requires_review=False,
                is_terminal=target in TERMINAL_STATUSES,
            )

        # Terminal state protection
        if current in TERMINAL_STATUSES:
            # Once CONCLUDED or CANCELLED, normal lifecycle events cannot overwrite it
            return TransitionResult(
                allowed=False,
                previous_status=current,
                new_status=current,
                event_type=event.event_type,
                reason=(
                    f"Transaction is in terminal status '{current.value}'. "
                    "Cannot be modified without an explicit trade correction."
                ),
                requires_review=True,
                is_terminal=True,
            )

        # Late-arriving message check
        event_dt = _parse_iso_timestamp(event.event_timestamp)
        last_dt = _parse_iso_timestamp(last_update_timestamp)
        if event_dt and last_dt and event_dt < last_dt:
            # Check if this transition would regress status
            target_status = _TRANSITION_RULES.get((current, event.event_type))
            if target_status is not None:
                current_rank = _STATUS_PROGRESSION_RANK.get(current, 0)
                target_rank = _STATUS_PROGRESSION_RANK.get(target_status, 0)
                if target_rank < current_rank:
                    return TransitionResult(
                        allowed=False,
                        previous_status=current,
                        new_status=current,
                        event_type=event.event_type,
                        reason=(
                            f"Late-arriving message (event timestamp {event.event_timestamp} < "
                            f"last update {last_update_timestamp}) cannot regress status from "
                            f"'{current.value}' to '{target_status.value}'."
                        ),
                        requires_review=True,
                        is_terminal=False,
                    )

        # Normal transition lookup
        key = (current, event.event_type)
        if key not in _TRANSITION_RULES:
            return TransitionResult(
                allowed=False,
                previous_status=current,
                new_status=current,
                event_type=event.event_type,
                reason=(
                    f"Unsupported transition from status '{current.value}' "
                    f"via event '{event.event_type.value}'."
                ),
                requires_review=True,
                is_terminal=current in TERMINAL_STATUSES,
            )

        new_status = _TRANSITION_RULES[key]
        return TransitionResult(
            allowed=True,
            previous_status=current,
            new_status=new_status,
            event_type=event.event_type,
            reason=f"Transition from {current.value} to {new_status.value} accepted via {event.event_type.value}.",
            requires_review=event.requires_review,
            is_terminal=new_status in TERMINAL_STATUSES,
        )


# --------------------------------------------------------------------------
# 4. LINGUISTIC / SEMANTIC EVENT CLASSIFIER
# --------------------------------------------------------------------------

# Hinglish & English confirmation expressions
_EXPLICIT_CONFIRMATION_PATTERNS = [
    re.compile(r"\b(?:deal\s+done|done\s+deal|trade\s+done|deal\s+confirmed|trade\s+confirmed)\b", re.I),
    re.compile(r"\b(?:booked|confirmed|executed)\b", re.I),
    re.compile(r"\b(?:done\s+hai|ho\s+gaya|sauda\s+done|pakka\s+hai|deal\s+pakki)\b", re.I),
    re.compile(r"\b(?:done\s*@|done\s+at)\s*\d", re.I),
    re.compile(r"\b(?:done\s+for)\s*\d", re.I),
]

# Vague confirmation words that REQUIRE strong context (never confirm alone)
_VAGUE_CONFIRMATION_WORDS = {
    "done", "ok", "okay", "yes", "deal", "ho gaya", "theek hai", "accha", "karo", "done hai"
}

# Cancellation expressions
_EXPLICIT_CANCELLATION_PATTERNS = [
    re.compile(r"\b(?:deal\s+cancel(?:led|ed)?|trade\s+cancel(?:led|ed)?|sauda\s+cancel)\b", re.I),
    re.compile(r"\b(?:cancelled|canceled)\b", re.I),
    re.compile(r"\b(?:cancel\s+(?:it|this|deal|trade|sauda|order|offer|bid))\b", re.I),
    re.compile(r"\b(?:cancel\s+kar\s*(?:do|karo|de|dena))\b", re.I),
    re.compile(r"\b(?:rehne\s+do|nahi\s+chahiye|drop\s+it|withdraw|discard)\b", re.I),
    re.compile(r"\b(?:no\s+longer\s+available|off\s+the\s+table)\b", re.I),
    re.compile(r"^\s*cancel\s*$", re.I),
]

# On-hold expressions
_ON_HOLD_PATTERNS = [
    re.compile(r"\b(?:on\s+hold|hold\s+on|hold\s+it|keep\s+on\s+hold)\b", re.I),
    re.compile(r"\b(?:hold\s+pe\s+rakho|hold\s+karo|ruko|wait)\b", re.I),
    re.compile(r"\b(?:let\s+you\s+know|checking|will\s+revert|pause)\b", re.I),
]

# Awaiting confirmation
_AWAITING_CONFIRMATION_PATTERNS = [
    re.compile(r"\b(?:confirm\s+pls|confirm\s+please|please\s+confirm|awaiting\s+confirm)\b", re.I),
    re.compile(r"\b(?:send\s+confirm|confirmation\s+pending)\b", re.I),
]

# Negotiation patterns
_NEGOTIATION_PATTERNS = [
    re.compile(r"\b(?:counter|counter\s+offer|revised|bhav|rate)\b", re.I),
    re.compile(r"\b(?:can\s+do\s*@|best\s*@|level\s*@|lvl\s*@)\b", re.I),
    re.compile(r"\b(?:too\s+high|too\s+low|kam\s+karo|tighten)\b", re.I),
    re.compile(r"\b(?:bid|offer)\s*[:=]?\s*\d+(?:\.\d+)?\s*%", re.I),
]

# Inquiry patterns
_INQUIRY_PATTERNS = [
    re.compile(r"\b(?:offer\s+pls|offer\s+please|bid\s+pls|bid\s+please)\b", re.I),
    re.compile(r"\b(?:any\s+bids|any\s+offers|need\s+quote|axe|looking\s+for)\b", re.I),
    re.compile(r"\b(?:lena\s+hai|dena\s+hai|anyone\s+buying|anyone\s+selling)\b", re.I),
]

# Correction patterns
_CORRECTION_PATTERNS = [
    re.compile(r"\b(?:correction|correcting|amended|amendment|corrected)\s*[:\-]", re.I),
    re.compile(r"\b(?:wrong\s+(?:rate|yield|quantity|isin)|correction\s+in)\b", re.I),
]


def classify_message_lifecycle_event(
    message_text: str,
    message_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    chat_key: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    candidate_count: int = 1,
    has_trade_terms: bool = False,
) -> LifecycleEvent:
    """
    Deterministically classify a message into a lifecycle event.

    Follows critical evidence rules:
    - A trade becomes concluded only when explicit trade confirmation/execution evidence is satisfied.
    - Negotiation, quotes, axes, inquiries are not proof of execution.
    - "done", "ok", "yes", "booked", "confirmed", "ho gaya", "done hai" are interpreted in context.
    - No keyword alone automatically completes a trade.
    - A vague acknowledgment with multiple candidate trades must NOT complete a trade (flagged as AMBIGUOUS).
    """
    cleaned = (message_text or "").strip()
    words = cleaned.lower().split()
    is_short = len(words) <= 3

    # Check for corrections first
    for pat in _CORRECTION_PATTERNS:
        match = pat.search(cleaned)
        if match:
            return LifecycleEvent(
                event_type=LifecycleEventType.TRADE_CORRECTED,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.9,
                requires_review=False,
                is_correction=True,
                correction_reason=f"Correction signal detected: '{match.group(0)}'",
            )

    # Check explicit cancellation
    for pat in _EXPLICIT_CANCELLATION_PATTERNS:
        match = pat.search(cleaned)
        if match:
            return LifecycleEvent(
                event_type=LifecycleEventType.TRADE_CANCELLED,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.95,
                requires_review=False,
            )

    # Check on hold
    for pat in _ON_HOLD_PATTERNS:
        match = pat.search(cleaned)
        if match:
            return LifecycleEvent(
                event_type=LifecycleEventType.TRADE_ON_HOLD,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.90,
                requires_review=False,
            )

    # Check awaiting confirmation
    for pat in _AWAITING_CONFIRMATION_PATTERNS:
        match = pat.search(cleaned)
        if match:
            return LifecycleEvent(
                event_type=LifecycleEventType.AWAITING_CONFIRMATION,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.90,
                requires_review=False,
            )

    # Check explicit trade confirmation patterns
    for pat in _EXPLICIT_CONFIRMATION_PATTERNS:
        match = pat.search(cleaned)
        if match:
            # If multiple candidates exist and no reply-to or explicit ref, flag ambiguous!
            if candidate_count > 1 and not reply_to_message_id:
                return LifecycleEvent(
                    event_type=LifecycleEventType.AMBIGUOUS,
                    message_id=message_id,
                    chat_id=chat_id,
                    chat_key=chat_key,
                    reply_to_message_id=reply_to_message_id,
                    event_timestamp=timestamp,
                    evidence_text=cleaned,
                    confidence_score=0.5,
                    requires_review=True,
                )
            return LifecycleEvent(
                event_type=LifecycleEventType.DEAL_CONFIRMED,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.95,
                requires_review=False,
            )

    # Vague acknowledgment ("done", "ok", "yes", "ho gaya", etc.)
    cleaned_lower = cleaned.lower()
    is_vague = any(
        cleaned_lower == vague or cleaned_lower.startswith(f"{vague} ") or cleaned_lower.endswith(f" {vague}")
        for vague in _VAGUE_CONFIRMATION_WORDS
    )
    if is_vague:
        # Critical evidence rule:
        # A vague acknowledgment MUST NOT complete a transaction when multiple candidate trades exist
        # or when there is no reply link and no explicit trade terms.
        if candidate_count > 1:
            return LifecycleEvent(
                event_type=LifecycleEventType.AMBIGUOUS,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.4,
                requires_review=True,
            )
        if reply_to_message_id or has_trade_terms:
            # Linked reply or accompanied by terms -> deal confirmed
            return LifecycleEvent(
                event_type=LifecycleEventType.DEAL_CONFIRMED,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.85,
                requires_review=False,
            )
        # Without reply linkage or terms, vague word alone is ambiguous
        return LifecycleEvent(
            event_type=LifecycleEventType.AMBIGUOUS,
            message_id=message_id,
            chat_id=chat_id,
            chat_key=chat_key,
            reply_to_message_id=reply_to_message_id,
            event_timestamp=timestamp,
            evidence_text=cleaned,
            confidence_score=0.45,
            requires_review=True,
        )

    # Check negotiation
    for pat in _NEGOTIATION_PATTERNS:
        match = pat.search(cleaned)
        if match:
            return LifecycleEvent(
                event_type=LifecycleEventType.NEGOTIATION,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.85,
                requires_review=False,
            )

    # Check inquiry
    for pat in _INQUIRY_PATTERNS:
        match = pat.search(cleaned)
        if match:
            return LifecycleEvent(
                event_type=LifecycleEventType.INQUIRY,
                message_id=message_id,
                chat_id=chat_id,
                chat_key=chat_key,
                reply_to_message_id=reply_to_message_id,
                event_timestamp=timestamp,
                evidence_text=cleaned,
                confidence_score=0.85,
                requires_review=False,
            )

    # Default: NO_LIFECYCLE_CHANGE
    return LifecycleEvent(
        event_type=LifecycleEventType.NO_LIFECYCLE_CHANGE,
        message_id=message_id,
        chat_id=chat_id,
        chat_key=chat_key,
        reply_to_message_id=reply_to_message_id,
        event_timestamp=timestamp,
        evidence_text=cleaned,
        confidence_score=1.0,
        requires_review=False,
    )
