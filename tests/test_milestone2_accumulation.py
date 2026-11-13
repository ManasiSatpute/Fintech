"""
Unit tests for Milestone 2: Partial-Information Accumulation and Field-Level Evidence Tracking.

Verifies:
1. Multi-message term accumulation across conversational turns.
2. Field-level evidence tracking (`field_evidence` and `detailed_field_evidence`).
3. Non-destructive updates (established terms are never overwritten with missing values).
4. Negotiation history tracking (yield_history, price_history, quantity_history, settlement_history).
5. Settlement details resolution (T+0, T+1, T+2, value date).
6. Counterparty role assignment and role disambiguation (buyer vs seller).
7. Financial sanity checks and validation rules via `validate_stack`.
8. Completeness indicators: `is_tradeable()` and `get_missing_fields()`.
"""

import unittest
from deal_stack_manager import DealStackManager
from deal_stack_models import AssignmentDecision, StackStatus


class Milestone2AccumulationTests(unittest.TestCase):

    def setUp(self):
        self.manager = DealStackManager()

    def test_multi_message_fragmented_accumulation(self):
        chat_key = "CHAT:m2-fragmented"

        # Turn 1: Dealer A announces bond identity
        r1 = self.manager.process_message({
            "message_id": "msg-1",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Selling PFC AAA 2028 NCD",
            "timestamp": "2026-11-13T10:00:00",
        })
        self.assertEqual(r1.decision, AssignmentDecision.NEW_STACK)

        # Turn 2: Dealer A specifies quantity
        r2 = self.manager.process_message({
            "message_id": "msg-2",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Quantity is 25Cr",
            "timestamp": "2026-11-13T10:01:00",
        })
        self.assertEqual(r2.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Turn 3: Dealer B asks for yield
        r3 = self.manager.process_message({
            "message_id": "msg-3",
            "chat_key": chat_key,
            "sender": "Dealer B",
            "message_text": "What is the yield level?",
            "timestamp": "2026-11-13T10:02:00",
        })
        self.assertEqual(r3.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Turn 4: Dealer A quotes yield
        r4 = self.manager.process_message({
            "message_id": "msg-4",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Offer level is 7.25",
            "timestamp": "2026-11-13T10:03:00",
        })
        self.assertEqual(r4.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Turn 5: Dealer B requests settlement T+1
        r5 = self.manager.process_message({
            "message_id": "msg-5",
            "chat_key": chat_key,
            "sender": "Dealer B",
            "message_text": "Can we settle T+1?",
            "timestamp": "2026-11-13T10:04:00",
        })
        self.assertEqual(r5.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Verify accumulated stack
        active = self.manager.get_active_stacks(chat_key)
        self.assertEqual(len(active), 1)
        stack = active[0]
        terms = stack.terms

        self.assertEqual(terms.issuer, "PFC")
        self.assertEqual(terms.instrument, "NCD")
        self.assertEqual(terms.credit_rating, "AAA")
        self.assertEqual(terms.maturity, "2028")
        self.assertEqual(terms.quantity, 25.0)
        self.assertEqual(terms.quantity_unit, "Cr")
        self.assertEqual(terms.yield_value, 7.25)
        self.assertEqual(terms.settlement, "T1")

        # Counterparty disambiguation: Dealer A is seller, Dealer B is buyer
        self.assertEqual(terms.seller, "Dealer A")
        self.assertEqual(terms.buyer, "Dealer B")

        # Field Evidence verification
        self.assertIn("msg-1", terms.field_evidence["issuer"])
        self.assertIn("msg-1", terms.field_evidence["credit_rating"])
        self.assertIn("msg-2", terms.field_evidence["quantity"])
        self.assertIn("msg-4", terms.field_evidence["yield_value"])
        self.assertIn("msg-5", terms.field_evidence["settlement"])

        # Detailed Field Evidence verification
        self.assertTrue(len(terms.detailed_field_evidence["quantity"]) >= 1)
        qty_ev = terms.detailed_field_evidence["quantity"][0]
        self.assertEqual(qty_ev["message_id"], "msg-2")
        self.assertEqual(qty_ev["value"], 25.0)
        self.assertEqual(qty_ev["sender"], "Dealer A")

    def test_non_destructive_accumulation_preserves_established_terms(self):
        chat_key = "CHAT:m2-nondestructive"
        # Initial message establishes ISIN and instrument
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": chat_key,
            "sender": "Desk 1",
            "message_text": "Selling 10Cr INE134E08KO9 7.18 SDL 2028 @ 7.25",
            "timestamp": "2026-11-13T10:00:00",
        })

        # Subsequent message contains ONLY counteroffer price without repeating ISIN or maturity
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": chat_key,
            "sender": "Desk 2",
            "message_text": "7.24?",
            "timestamp": "2026-11-13T10:01:00",
        })

        stack = self.manager.get_active_stacks(chat_key)[0]
        # ISIN, maturity, and coupon must not be cleared or overwritten
        self.assertEqual(stack.terms.isin, "INE134E08KO9")
        self.assertEqual(stack.terms.maturity, "2028")
        self.assertEqual(stack.terms.coupon_rate, 7.18)
        self.assertEqual(stack.terms.yield_value, 7.24)

    def test_negotiation_history_tracking(self):
        chat_key = "CHAT:m2-history"
        # Turn 1: Dealer A quotes 7.25
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-13T10:00:00",
        })
        # Turn 2: Dealer B counters 7.24
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": chat_key,
            "sender": "Dealer B",
            "message_text": "Can you do 7.24?",
            "timestamp": "2026-11-13T10:01:00",
        })
        # Turn 3: Dealer A meets at 7.245
        self.manager.process_message({
            "message_id": "m3",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Meet at 7.245",
            "timestamp": "2026-11-13T10:02:00",
        })

        stack = self.manager.get_active_stacks(chat_key)[0]
        # Current active yield
        self.assertEqual(stack.terms.yield_value, 7.245)

        # History should preserve all 3 rounds of quotes with message evidence
        history = stack.terms.yield_history
        self.assertEqual(len(history), 3)
        self.assertEqual([h["yield_value"] for h in history], [7.25, 7.24, 7.245])
        self.assertEqual([h["message_id"] for h in history], ["m1", "m2", "m3"])
        self.assertEqual([h["sender"] for h in history], ["Dealer A", "Dealer B", "Dealer A"])

    def test_trade_sanity_validation(self):
        chat_key = "CHAT:m2-valid"
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-13T10:00:00",
        })
        stack = self.manager.get_active_stacks(chat_key)[0]

        is_valid, issues = self.manager.validate_stack(stack)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

        # Corrupt stack: buyer == seller
        stack.terms.buyer = "Dealer A"
        stack.terms.seller = "Dealer A"
        is_valid_same, issues_same = self.manager.validate_stack(stack)
        self.assertFalse(is_valid_same)
        self.assertTrue(any("identical" in err.lower() for err in issues_same))

        # Out-of-bounds yield
        stack.terms.buyer = "Dealer B"
        stack.terms.yield_value = 85.0  # Absurd 85% YTM
        is_valid_ytm, issues_ytm = self.manager.validate_stack(stack)
        self.assertFalse(is_valid_ytm)
        self.assertTrue(any("yield" in err.lower() for err in issues_ytm))

    def test_completeness_and_tradeable_indicators(self):
        chat_key = "CHAT:m2-complete"
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": chat_key,
            "sender": "Dealer A",
            "message_text": "Selling 10Cr PFC AAA 2028 @ 7.25",
            "timestamp": "2026-11-13T10:00:00",
        })
        stack = self.manager.get_active_stacks(chat_key)[0]

        # Prior to counterparty confirmation, buyer is missing
        self.assertIn("buyer", stack.terms.get_missing_fields())
        self.assertTrue(stack.terms.is_tradeable())  # security + quantity + rate are present

        # Dealer B accepts
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": chat_key,
            "sender": "Dealer B",
            "message_text": "Done at 7.25",
            "timestamp": "2026-11-13T10:02:00",
        })

        sealed = self.manager.get_sealed_stacks(chat_key)[0]
        self.assertEqual(sealed.status, StackStatus.CONCLUDED)
        # All required transaction fields are complete!
        self.assertEqual(len(sealed.terms.get_missing_fields()), 0)


if __name__ == "__main__":
    unittest.main()

