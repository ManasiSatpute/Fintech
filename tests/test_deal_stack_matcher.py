"""
Unit tests for Dynamic Deal Stack Models and Message-to-Stack Matcher.
"""

import unittest
from datetime import datetime

import config
from deal_stack_models import (
    AssignmentDecision,
    DealStack,
    DealTerms,
    EvidenceTier,
    MessageBondFeatures,
    SealReason,
    StackMessage,
    StackState,
    StackStatus,
)
from deal_stack_matcher import (
    compute_subject_fingerprint,
    extract_bond_features,
    match_message_to_stacks,
    score_stack_candidate,
)


def _make_stack(
    stack_id: str,
    instrument: str = "SDL",
    maturity: str = "2028",
    isin: str = None,
    issuer: str = None,
    coupon: float = 7.18,
    yield_val: float = 7.25,
    quantity: float = 5.0,
    counterparties: list = None,
    messages: list = None,
) -> DealStack:
    now_iso = datetime.now().isoformat()
    counterparties = counterparties or ["Dealer A", "Dealer B"]
    terms = DealTerms(
        isin=isin,
        issuer=issuer,
        instrument=instrument,
        maturity=maturity,
        coupon_rate=coupon,
        quantity=quantity,
        yield_value=yield_val,
        counterparties=counterparties,
    )
    stack_msgs = []
    for i, m in enumerate(messages or []):
        if isinstance(m, StackMessage):
            stack_msgs.append(m)
        else:
            stack_msgs.append(
                StackMessage(
                    message_id=m.get("message_id", f"msg-{i}"),
                    sender=m.get("sender", "Dealer A"),
                    message_text=m.get("message_text", "trade message"),
                    timestamp=m.get("timestamp", now_iso),
                    assignment_reason="test",
                    assignment_confidence=1.0,
                )
            )

    return DealStack(
        stack_id=stack_id,
        chat_key="CHAT:test-group",
        instrument=instrument,
        maturity=maturity,
        isin=isin,
        issuer=issuer,
        coupon_rate=coupon,
        terms=terms,
        status=StackStatus.PENDING,
        stack_state=StackState.ACTIVE,
        opened_at=now_iso,
        last_activity_at=now_iso,
        messages=stack_msgs,
    )


class DealStackMatcherTests(unittest.TestCase):

    def test_pure_smalltalk_excluded_as_noise(self):
        msg = {
            "message_id": "m1",
            "sender": "Dealer A",
            "message_text": "Good morning all",
            "timestamp": "2026-11-03T10:00:00",
        }
        res = match_message_to_stacks(msg, active_stacks=[])
        self.assertEqual(res.decision, AssignmentDecision.NOISE_EXCLUDED)
        self.assertFalse(res.requires_human_review)

    def test_opening_quote_creates_new_stack_when_no_active_stacks(self):
        msg = {
            "message_id": "m2",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr 7.18 SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        }
        res = match_message_to_stacks(msg, active_stacks=[])
        self.assertEqual(res.decision, AssignmentDecision.NEW_STACK)
        self.assertGreaterEqual(res.confidence_score, 0.85)
        self.assertFalse(res.requires_human_review)

    def test_two_different_bonds_negotiated_simultaneously(self):
        # Stack 1: SDL 2028 (Dealer A selling 5Cr)
        stack1 = _make_stack(
            stack_id="stack-sdl-2028",
            instrument="SDL",
            maturity="2028",
            coupon=7.18,
            yield_val=7.25,
            quantity=5.0,
            counterparties=["Dealer A"],
            messages=[{
                "message_id": "m-sdl-1",
                "sender": "Dealer A",
                "message_text": "Selling 5Cr 7.18 SDL 2028",
                "timestamp": "2026-11-03T10:00:00",
            }],
        )

        # Message introducing Deal B: GS 2027
        msg_deal_b = {
            "message_id": "m-gs-1",
            "sender": "Dealer C",
            "message_text": "Looking to buy 2Cr 6.79 GS 2027",
            "timestamp": "2026-11-03T10:02:00",
        }

        # Must recognize distinct bond (different instrument and maturity) and create NEW_STACK
        res = match_message_to_stacks(msg_deal_b, active_stacks=[stack1])
        self.assertEqual(res.decision, AssignmentDecision.NEW_STACK)
        # Verify hard conflict was recorded against stack 1
        self.assertTrue(res.candidate_matches[0].is_conflicting_bond)
        self.assertIn("Differing instrument", res.candidate_matches[0].conflict_reason)

    def test_price_counteroffer_stays_in_existing_stack(self):
        stack1 = _make_stack(
            stack_id="stack-1",
            instrument="SDL",
            maturity="2028",
            yield_val=7.25,
            counterparties=["Dealer A", "Dealer B"],
            messages=[
                {"message_id": "m1", "sender": "Dealer A", "message_text": "Selling 5Cr SDL 2028 @ 7.25"},
                {"message_id": "m2", "sender": "Dealer B", "message_text": "Can you do 7.24?"},
            ],
        )

        counter_msg = {
            "message_id": "m3",
            "sender": "Dealer A",
            "message_text": "Can do @ 7.245",
            "timestamp": "2026-11-03T10:03:00",
        }
        res = match_message_to_stacks(counter_msg, active_stacks=[stack1])
        self.assertEqual(res.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res.assigned_stack_id, "stack-1")
        self.assertFalse(res.requires_human_review)

    def test_exact_isin_match_routes_to_matching_stack(self):
        stack1 = _make_stack(
            stack_id="stack-isin-1",
            isin="IN0020200001",
            yield_val=7.10,
            counterparties=["Desk 1", "Desk 2"],
            messages=[{"message_id": "m1", "sender": "Desk 1", "message_text": "IN0020200001 offer @ 7.10"}],
        )
        stack2 = _make_stack(
            stack_id="stack-isin-2",
            isin="IN0020210002",
            yield_val=6.85,
            counterparties=["Desk 3", "Desk 4"],
            messages=[{"message_id": "m2", "sender": "Desk 3", "message_text": "IN0020210002 bid @ 6.85"}],
        )

        msg = {
            "message_id": "m3",
            "sender": "Desk 2",
            "message_text": "Taking IN0020200001 at 7.10 done",
            "timestamp": "2026-11-03T10:05:00",
        }
        res = match_message_to_stacks(msg, active_stacks=[stack1, stack2])
        self.assertEqual(res.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res.assigned_stack_id, "stack-isin-1")

    def test_same_issuer_different_maturity_creates_new_stack(self):
        # Stack 1: GS 2027
        stack1 = _make_stack(
            stack_id="stack-gs-2027",
            instrument="GS",
            maturity="2027",
            counterparties=["Trader A"],
            messages=[{"message_id": "m1", "sender": "Trader A", "message_text": "Selling 10Cr GS 2027"}],
        )

        # Message for GS 2030 (same issuer/instrument, but different maturity year)
        msg_diff_mat = {
            "message_id": "m2",
            "sender": "Trader B",
            "message_text": "Any bids for 5Cr GS 2030?",
            "timestamp": "2026-11-03T10:06:00",
        }
        res = match_message_to_stacks(msg_diff_mat, active_stacks=[stack1])
        self.assertEqual(res.decision, AssignmentDecision.NEW_STACK)
        self.assertTrue(res.candidate_matches[0].is_conflicting_bond)
        self.assertIn("Differing maturity", res.candidate_matches[0].conflict_reason)

    def test_independent_negotiation_same_isin_from_new_counterparty(self):
        # Stack 1 is between Dealer A and Dealer B
        stack1 = _make_stack(
            stack_id="stack-1",
            isin="IN0020200001",
            yield_val=7.15,
            quantity=5.0,
            counterparties=["Dealer A", "Dealer B"],
            messages=[
                {"message_id": "m1", "sender": "Dealer A", "message_text": "Selling 5Cr IN0020200001 @ 7.15"},
                {"message_id": "m2", "sender": "Dealer B", "message_text": "Bid 7.14"},
            ],
        )

        # Dealer C arrives and posts an independent opening offer for 10Cr of the same ISIN
        msg_dealer_c = {
            "message_id": "m3",
            "sender": "Dealer C",
            "message_text": "I have 10Cr IN0020200001 to sell at 7.16",
            "timestamp": "2026-11-03T10:10:00",
        }
        res = match_message_to_stacks(msg_dealer_c, active_stacks=[stack1])
        # Should recognize Dealer C is not continuing A-B negotiation, but opening a new independent deal
        self.assertEqual(res.decision, AssignmentDecision.NEW_STACK)
        self.assertIn("Independent negotiation", res.assignment_reason)

    def test_ambiguous_done_without_context_flags_unresolved(self):
        stack1 = _make_stack(stack_id="stack-1", yield_val=7.25, counterparties=["A", "B"])
        stack2 = _make_stack(stack_id="stack-2", yield_val=6.80, counterparties=["C", "D"])

        # Ambiguous "done" with no reply-to and no bond terms from an unknown party
        msg = {
            "message_id": "m-ambig",
            "sender": "Unknown Trader",
            "message_text": "Done",
            "timestamp": "2026-11-03T10:15:00",
        }
        res = match_message_to_stacks(msg, active_stacks=[stack1, stack2])
        self.assertEqual(res.decision, AssignmentDecision.UNRESOLVED)
        self.assertTrue(res.requires_human_review)

    def test_reply_to_older_message_routes_to_correct_stack(self):
        stack1 = _make_stack(
            stack_id="stack-1",
            instrument="SDL",
            maturity="2028",
            messages=[{"message_id": "target-msg-id", "sender": "Dealer A", "message_text": "Selling 5Cr SDL 2028"}],
        )
        stack2 = _make_stack(
            stack_id="stack-2",
            instrument="GS",
            maturity="2027",
            messages=[{"message_id": "other-msg-id", "sender": "Dealer C", "message_text": "GS 2027"}],
        )

        reply_msg = {
            "message_id": "m-reply",
            "sender": "Dealer B",
            "message_text": "7.24 done",
            "reply_to_message_id": "target-msg-id",
            "timestamp": "2026-11-03T10:20:00",
        }
        res = match_message_to_stacks(reply_msg, active_stacks=[stack1, stack2])
        self.assertEqual(res.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res.assigned_stack_id, "stack-1")
        self.assertEqual(res.confidence_score, 1.0)

    def test_partial_information_message_accumulation(self):
        # Stack 1 opened with partial info
        stack1 = _make_stack(
            stack_id="stack-partial",
            instrument="SDL",
            maturity="2028",
            counterparties=["Dealer A"],
            messages=[{"message_id": "m1", "sender": "Dealer A", "message_text": "I have SDL 2028 to sell"}],
        )

        # Message 2 adds quantity
        msg_qty = {
            "message_id": "m2",
            "sender": "Dealer A",
            "message_text": "Quantity is 5Cr",
            "timestamp": "2026-11-03T10:01:00",
        }
        res_qty = match_message_to_stacks(msg_qty, active_stacks=[stack1])
        self.assertEqual(res_qty.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res_qty.assigned_stack_id, "stack-partial")

    def test_missing_isin_but_sufficient_issuer_maturity_matches(self):
        stack1 = _make_stack(
            stack_id="stack-oxyzo",
            issuer="Oxyzo",
            instrument="NCD",
            maturity="2026",
            counterparties=["Dealer A"],
            messages=[{"message_id": "m1", "sender": "Dealer A", "message_text": "Oxyzo NCD 2026 axe"}],
        )

        msg = {
            "message_id": "m2",
            "sender": "Dealer B",
            "message_text": "Can take 5Cr Oxyzo 2026 @ 8.25%",
            "timestamp": "2026-11-03T10:02:00",
        }
        res = match_message_to_stacks(msg, active_stacks=[stack1])
        self.assertEqual(res.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res.assigned_stack_id, "stack-oxyzo")

    def test_reply_to_historical_stack(self):
        # Historical (sealed) stack
        hist_stack = _make_stack(
            stack_id="hist-stack-1",
            instrument="GS",
            maturity="2027",
            messages=[{"message_id": "old-msg-1", "sender": "Dealer A", "message_text": "Closed GS 2027"}],
        )
        hist_stack.stack_state = StackState.SEALED
        hist_stack.status = StackStatus.CONCLUDED
        hist_stack.seal_reason = SealReason.EXPLICIT_CONCLUSION

        reply_msg = {
            "message_id": "m-late-reply",
            "sender": "Dealer B",
            "message_text": "Confirming settlement for old trade",
            "reply_to_message_id": "old-msg-1",
            "timestamp": "2026-11-03T11:00:00",
        }
        res = match_message_to_stacks(reply_msg, active_stacks=[], historical_stacks=[hist_stack])
        self.assertEqual(res.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res.assigned_stack_id, "hist-stack-1")
        self.assertGreaterEqual(res.confidence_score, 0.90)

    def test_multiple_deals_from_same_issuer(self):
        # NHAI 2027 vs NHAI 2032
        stack_nhai_2027 = _make_stack(
            stack_id="stack-nhai-27",
            issuer="NHAI",
            maturity="2027",
            counterparties=["Desk 1"],
            messages=[{"message_id": "m1", "sender": "Desk 1", "message_text": "NHAI 2027 available"}],
        )

        msg_nhai_2032 = {
            "message_id": "m2",
            "sender": "Desk 2",
            "message_text": "Selling 10Cr NHAI 2032 @ 7.45",
            "timestamp": "2026-11-03T10:05:00",
        }
        res = match_message_to_stacks(msg_nhai_2032, active_stacks=[stack_nhai_2027])
        self.assertEqual(res.decision, AssignmentDecision.NEW_STACK)
        self.assertTrue(res.candidate_matches[0].is_conflicting_bond)
        self.assertIn("Differing maturity", res.candidate_matches[0].conflict_reason)

    def test_price_change_alone_does_not_create_new_stack(self):
        stack = _make_stack(
            stack_id="stack-price",
            instrument="SDL",
            maturity="2028",
            yield_val=7.25,
            counterparties=["Dealer A"],
            messages=[{"message_id": "m1", "sender": "Dealer A", "message_text": "Selling 5Cr SDL 2028 @ 7.25"}],
        )

        # Same dealer revises price to 7.24
        msg_revise = {
            "message_id": "m2",
            "sender": "Dealer A",
            "message_text": "Revised level: 7.24",
            "timestamp": "2026-11-03T10:02:00",
        }
        res = match_message_to_stacks(msg_revise, active_stacks=[stack])
        self.assertEqual(res.decision, AssignmentDecision.ASSIGN_EXISTING)
        self.assertEqual(res.assigned_stack_id, "stack-price")

    def test_configurable_thresholds_exist_in_config(self):
        self.assertTrue(hasattr(config, "STACK_HIGH_CONFIDENCE_THRESHOLD"))
        self.assertTrue(hasattr(config, "STACK_NEW_NEGOTIATION_THRESHOLD"))
        self.assertTrue(hasattr(config, "STACK_AMBIGUITY_DELTA_THRESHOLD"))
        self.assertTrue(hasattr(config, "STACK_TIME_DECAY_HOURS"))
        self.assertGreater(config.STACK_HIGH_CONFIDENCE_THRESHOLD, config.STACK_NEW_NEGOTIATION_THRESHOLD)


class DealStackModelsTests(unittest.TestCase):

    def test_strictly_three_business_statuses(self):
        self.assertEqual(set(StackStatus), {StackStatus.PENDING, StackStatus.CONCLUDED, StackStatus.REJECTED})
        self.assertEqual(len(StackStatus), 3)

    def test_subject_fingerprint_generation(self):
        terms_isin = DealTerms(isin="IN0020200001")
        self.assertEqual(compute_subject_fingerprint(terms_isin), "ISIN:IN0020200001")

        terms_bond = DealTerms(instrument="SDL", maturity="2028", coupon_rate=7.18)
        self.assertEqual(compute_subject_fingerprint(terms_bond), "BOND:SDL:2028:7.18%")


if __name__ == "__main__":
    unittest.main()
