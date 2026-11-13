"""
Unit tests for the lifecycle state transition engine and event models.
"""

import unittest
from lifecycle import (
    LifecycleEngine,
    LifecycleEvent,
    LifecycleEventType,
    LifecycleStatus,
    classify_message_lifecycle_event,
)


class LifecycleStateEngineTests(unittest.TestCase):
    def test_inquiry_followed_by_negotiation(self):
        event = LifecycleEvent(
            event_type=LifecycleEventType.NEGOTIATION,
            evidence_text="Can do @ 7.85%",
            confidence_score=0.9,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.INITIATED, event)
        self.assertTrue(res.allowed)
        self.assertEqual(res.previous_status, LifecycleStatus.INITIATED)
        self.assertEqual(res.new_status, LifecycleStatus.NEGOTIATING)
        self.assertFalse(res.requires_review)

    def test_negotiation_followed_by_pending_confirmation(self):
        event = LifecycleEvent(
            event_type=LifecycleEventType.AWAITING_CONFIRMATION,
            evidence_text="Please confirm 10 Cr @ 7.85%",
            confidence_score=0.9,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.NEGOTIATING, event)
        self.assertTrue(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.PENDING_CONFIRMATION)

    def test_explicit_confirmation_leading_to_concluded(self):
        event = LifecycleEvent(
            event_type=LifecycleEventType.DEAL_CONFIRMED,
            evidence_text="Deal confirmed for 10 Cr",
            confidence_score=0.95,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.PENDING_CONFIRMATION, event)
        self.assertTrue(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.CONCLUDED)
        self.assertTrue(res.is_terminal)

    def test_explicit_execution_leading_to_concluded(self):
        event = LifecycleEvent(
            event_type=LifecycleEventType.TRADE_EXECUTED,
            evidence_text="Trade executed at 7.80%",
            confidence_score=0.95,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.NEGOTIATING, event)
        self.assertTrue(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.CONCLUDED)
        self.assertTrue(res.is_terminal)

    def test_explicit_cancellation_leading_to_cancelled(self):
        for initial in (LifecycleStatus.INITIATED, LifecycleStatus.NEGOTIATING, LifecycleStatus.PENDING_CONFIRMATION):
            event = LifecycleEvent(
                event_type=LifecycleEventType.TRADE_CANCELLED,
                evidence_text="Deal cancelled by buyer",
                confidence_score=0.95,
            )
            res = LifecycleEngine.validate_transition(initial, event)
            self.assertTrue(res.allowed)
            self.assertEqual(res.new_status, LifecycleStatus.CANCELLED)
            self.assertTrue(res.is_terminal)

    def test_on_hold_transitions(self):
        event_hold = LifecycleEvent(
            event_type=LifecycleEventType.TRADE_ON_HOLD,
            evidence_text="Keep on hold for 10 mins",
        )
        res1 = LifecycleEngine.validate_transition(LifecycleStatus.NEGOTIATING, event_hold)
        self.assertTrue(res1.allowed)
        self.assertEqual(res1.new_status, LifecycleStatus.ON_HOLD)

        # Resume negotiation from on-hold
        event_resume = LifecycleEvent(
            event_type=LifecycleEventType.NEGOTIATION,
            evidence_text="Back online, counter at 7.82%",
        )
        res2 = LifecycleEngine.validate_transition(LifecycleStatus.ON_HOLD, event_resume)
        self.assertTrue(res2.allowed)
        self.assertEqual(res2.new_status, LifecycleStatus.NEGOTIATING)

    def test_concluded_terminal_state_protection(self):
        # Normal event cannot alter CONCLUDED
        event = LifecycleEvent(
            event_type=LifecycleEventType.NEGOTIATION,
            evidence_text="Counter offer @ 7.75%",
            confidence_score=0.9,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.CONCLUDED, event)
        self.assertFalse(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.CONCLUDED)
        self.assertTrue(res.requires_review)
        self.assertIn("terminal status", res.reason)

    def test_cancelled_terminal_state_protection(self):
        # Normal event cannot alter CANCELLED
        event = LifecycleEvent(
            event_type=LifecycleEventType.DEAL_CONFIRMED,
            evidence_text="Confirmed",
            confidence_score=0.9,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.CANCELLED, event)
        self.assertFalse(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.CANCELLED)
        self.assertTrue(res.requires_review)

    def test_explicit_correction_to_concluded_trade(self):
        # Correction without reason is rejected
        bad_correction = LifecycleEvent(
            event_type=LifecycleEventType.TRADE_CORRECTED,
            evidence_text="Correction",
            is_correction=True,
            correction_reason=None,
        )
        res_bad = LifecycleEngine.validate_transition(LifecycleStatus.CONCLUDED, bad_correction)
        self.assertFalse(res_bad.allowed)

        # Correction with reason is accepted
        good_correction = LifecycleEvent(
            event_type=LifecycleEventType.TRADE_CORRECTED,
            evidence_text="Correction: yield amended to 7.82%",
            is_correction=True,
            correction_reason="Yield entered incorrectly in negotiation",
        )
        res_good = LifecycleEngine.validate_transition(LifecycleStatus.CONCLUDED, good_correction)
        self.assertTrue(res_good.allowed)
        self.assertIn("correction", res_good.reason.lower())

    def test_late_arriving_message_cannot_regress_state(self):
        # Concluded transaction with update at 10:30
        last_update = "2026-10-26T10:30:00"
        # Late message with earlier timestamp 10:15 attempting negotiation
        event = LifecycleEvent(
            event_type=LifecycleEventType.NEGOTIATION,
            event_timestamp="2026-10-26T10:15:00",
            evidence_text="Counter @ 7.85%",
        )
        res = LifecycleEngine.validate_transition(
            LifecycleStatus.PENDING_CONFIRMATION,
            event,
            last_update_timestamp=last_update,
        )
        self.assertFalse(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.PENDING_CONFIRMATION)
        self.assertIn("Late-arriving message", res.reason)

    def test_ambiguous_event_preserves_status_and_requires_review(self):
        event = LifecycleEvent(
            event_type=LifecycleEventType.AMBIGUOUS,
            evidence_text="ok",
            confidence_score=0.4,
            requires_review=True,
        )
        res = LifecycleEngine.validate_transition(LifecycleStatus.NEGOTIATING, event)
        self.assertTrue(res.allowed)
        self.assertEqual(res.new_status, LifecycleStatus.NEGOTIATING)
        self.assertTrue(res.requires_review)


class LifecycleClassificationTests(unittest.TestCase):
    def test_vague_done_without_context_is_ambiguous(self):
        event = classify_message_lifecycle_event("done", candidate_count=1)
        self.assertEqual(event.event_type, LifecycleEventType.AMBIGUOUS)
        self.assertTrue(event.requires_review)

    def test_vague_done_with_multiple_candidates_is_ambiguous(self):
        event = classify_message_lifecycle_event("done hai", candidate_count=3)
        self.assertEqual(event.event_type, LifecycleEventType.AMBIGUOUS)
        self.assertTrue(event.requires_review)

    def test_vague_ok_with_reply_link_is_confirmed(self):
        event = classify_message_lifecycle_event(
            "ok",
            reply_to_message_id="msg-123",
            candidate_count=1,
        )
        self.assertEqual(event.event_type, LifecycleEventType.DEAL_CONFIRMED)
        self.assertFalse(event.requires_review)

    def test_hinglish_confirmation_patterns(self):
        for msg in ("deal pakki", "sauda done", "ho gaya", "done hai"):
            event = classify_message_lifecycle_event(msg, reply_to_message_id="msg-1")
            self.assertEqual(event.event_type, LifecycleEventType.DEAL_CONFIRMED, f"Failed for '{msg}'")

    def test_hinglish_cancellation_patterns(self):
        for msg in ("cancel kar do", "nahi chahiye", "rehne do", "deal cancel"):
            event = classify_message_lifecycle_event(msg)
            self.assertEqual(event.event_type, LifecycleEventType.TRADE_CANCELLED, f"Failed for '{msg}'")

    def test_hinglish_on_hold_patterns(self):
        for msg in ("hold pe rakho", "hold karo", "ruko thoda"):
            event = classify_message_lifecycle_event(msg)
            self.assertEqual(event.event_type, LifecycleEventType.TRADE_ON_HOLD, f"Failed for '{msg}'")

    def test_correction_patterns(self):
        event = classify_message_lifecycle_event("Correction: rate is 7.82% not 7.85%")
        self.assertEqual(event.event_type, LifecycleEventType.TRADE_CORRECTED)
        self.assertTrue(event.is_correction)


if __name__ == "__main__":
    unittest.main()
