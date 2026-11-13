"""
Comprehensive Unit Tests for Dynamic Deal Stack Manager.

Verifies all 15 required scenarios:
1. Creating a new stack.
2. Assigning messages to existing stacks.
3. Maintaining two simultaneous negotiations.
4. Maintaining separate negotiations for the same ISIN.
5. Accumulating partial bond information.
6. Keeping price counteroffers in the same stack.
7. Handling ambiguous messages.
8. Excluding non-financial noise.
9. Explicit conclusion.
10. Explicit rejection.
11. Batch-end snapshots.
12. Continuing pending negotiations across uploads.
13. Preventing duplicate message assignments.
14. Handling late-arriving messages.
15. Ensuring one chat cannot contaminate another chat's stacks.
"""

import unittest
from datetime import datetime

from deal_stack_manager import DealStackManager
from deal_stack_models import (
    AssignmentDecision,
    DealStack,
    SealReason,
    StackState,
    StackStatus,
)


class DealStackManagerTests(unittest.TestCase):

    def setUp(self):
        self.manager = DealStackManager()

    # ----------------------------------------------------------------------
    # 1. Creating a new stack
    # ----------------------------------------------------------------------
    def test_create_new_stack(self):
        msg = {
            "message_id": "msg-1",
            "chat_key": "CHAT:group-alpha",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr 7.18 SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        }
        res = self.manager.process_message(msg)
        self.assertEqual(res.decision, AssignmentDecision.NEW_STACK)
        self.assertIsNotNone(res.assigned_stack_id)

        active = self.manager.get_active_stacks("CHAT:group-alpha")
        self.assertEqual(len(active), 1)
        stack = active[0]
        self.assertEqual(stack.stack_id, res.assigned_stack_id)
        self.assertEqual(stack.instrument, "SDL")
        self.assertEqual(stack.maturity, "2028")
        self.assertEqual(stack.terms.quantity, 5.0)
        self.assertEqual(stack.terms.yield_value, 7.25)
        self.assertEqual(stack.status, StackStatus.PENDING)
        self.assertEqual(stack.stack_state, StackState.ACTIVE)
        self.assertEqual(len(stack.messages), 1)

    # ----------------------------------------------------------------------
    # 2. Assigning messages to existing stacks
    # ----------------------------------------------------------------------
    def test_assigning_messages_to_existing_stack(self):
        self.manager.process_message({
            "message_id": "msg-1",
            "chat_key": "CHAT:group-alpha",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr 7.18 SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })

        # Message 2 from Dealer B asking for price/rate confirmation
        msg2 = {
            "message_id": "msg-2",
            "chat_key": "CHAT:group-alpha",
            "sender": "Dealer B",
            "message_text": "What is the price?",
            "timestamp": "2026-11-03T10:01:00",
        }
        res2 = self.manager.process_message(msg2)
        self.assertEqual(res2.decision, AssignmentDecision.ASSIGN_EXISTING)

        active = self.manager.get_active_stacks("CHAT:group-alpha")
        self.assertEqual(len(active), 1)
        self.assertEqual(len(active[0].messages), 2)
        self.assertEqual(active[0].messages[1].message_id, "msg-2")
        self.assertIn("Dealer B", active[0].terms.counterparties)

    # ----------------------------------------------------------------------
    # 3. Maintaining two simultaneous negotiations
    # ----------------------------------------------------------------------
    def test_maintaining_two_simultaneous_negotiations(self):
        # 10:00 - A: Selling 5Cr SDL 2028
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:otc-desk",
            "sender": "A",
            "message_text": "Selling 5Cr SDL 2028",
            "timestamp": "2026-11-03T10:00:00",
        })
        # 10:01 - B: What is the price?
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:otc-desk",
            "sender": "B",
            "message_text": "What is the price?",
            "timestamp": "2026-11-03T10:01:00",
        })
        # 10:02 - C: Looking to buy 2Cr GS 2027 (Independent simultaneous deal!)
        res_c = self.manager.process_message({
            "message_id": "m3",
            "chat_key": "CHAT:otc-desk",
            "sender": "C",
            "message_text": "Looking to buy 2Cr GS 2027",
            "timestamp": "2026-11-03T10:02:00",
        })
        # 10:03 - A: 7.25. (Belongs to Deal A)
        res_a = self.manager.process_message({
            "message_id": "m4",
            "chat_key": "CHAT:otc-desk",
            "sender": "A",
            "message_text": "7.25",
            "timestamp": "2026-11-03T10:03:00",
        })

        self.assertEqual(res_c.decision, AssignmentDecision.NEW_STACK)
        self.assertEqual(res_a.decision, AssignmentDecision.ASSIGN_EXISTING)

        active = self.manager.get_active_stacks("CHAT:otc-desk")
        self.assertEqual(len(active), 2)

        stack_sdl = next(s for s in active if s.instrument == "SDL")
        stack_gs = next(s for s in active if s.instrument == "GS")

        self.assertEqual(stack_sdl.maturity, "2028")
        self.assertEqual(stack_sdl.terms.quantity, 5.0)
        self.assertEqual(stack_sdl.terms.yield_value, 7.25)
        self.assertEqual([m.message_id for m in stack_sdl.messages], ["m1", "m2", "m4"])

        self.assertEqual(stack_gs.maturity, "2027")
        self.assertEqual(stack_gs.terms.quantity, 2.0)
        self.assertEqual([m.message_id for m in stack_gs.messages], ["m3"])

    # ----------------------------------------------------------------------
    # 4. Maintaining separate negotiations for the same ISIN
    # ----------------------------------------------------------------------
    def test_maintaining_separate_negotiations_for_same_isin(self):
        isin = "IN0020200001"
        # Deal 1: Dealer A and Dealer B negotiating ISIN IN0020200001 for 5Cr
        self.manager.process_message({
            "message_id": "d1-m1",
            "chat_key": "CHAT:bond-room",
            "sender": "Dealer A",
            "message_text": f"Selling 5Cr {isin} @ 7.15",
            "timestamp": "2026-11-03T10:00:00",
        })
        self.manager.process_message({
            "message_id": "d1-m2",
            "chat_key": "CHAT:bond-room",
            "sender": "Dealer B",
            "message_text": "Can you do 7.14?",
            "timestamp": "2026-11-03T10:01:00",
        })

        # Deal 2: Dealer C independently offers 10Cr of the same ISIN at 7.18
        res_d2 = self.manager.process_message({
            "message_id": "d2-m1",
            "chat_key": "CHAT:bond-room",
            "sender": "Dealer C",
            "message_text": f"I have 10Cr {isin} to sell at 7.18",
            "timestamp": "2026-11-03T10:05:00",
        })

        self.assertEqual(res_d2.decision, AssignmentDecision.NEW_STACK)

        active = self.manager.get_active_stacks("CHAT:bond-room")
        self.assertEqual(len(active), 2)
        self.assertNotEqual(active[0].stack_id, active[1].stack_id)
        self.assertEqual(active[0].isin, isin)
        self.assertEqual(active[1].isin, isin)

    # ----------------------------------------------------------------------
    # 5. Accumulating partial bond information across messages
    # ----------------------------------------------------------------------
    def test_accumulating_partial_bond_information(self):
        # Message 1: Instrument & maturity
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:group-partial",
            "sender": "Desk A",
            "message_text": "I have SDL 2028 to sell",
            "timestamp": "2026-11-03T10:00:00",
        })
        # Message 2: Quantity
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:group-partial",
            "sender": "Desk A",
            "message_text": "Quantity is 5Cr",
            "timestamp": "2026-11-03T10:01:00",
        })
        # Message 3: Level/yield
        self.manager.process_message({
            "message_id": "m3",
            "chat_key": "CHAT:group-partial",
            "sender": "Desk A",
            "message_text": "Offer level is 7.25",
            "timestamp": "2026-11-03T10:02:00",
        })

        active = self.manager.get_active_stacks("CHAT:group-partial")
        self.assertEqual(len(active), 1)
        stack = active[0]
        self.assertEqual(stack.instrument, "SDL")
        self.assertEqual(stack.maturity, "2028")
        self.assertEqual(stack.terms.quantity, 5.0)
        self.assertEqual(stack.terms.yield_value, 7.25)
        self.assertEqual(len(stack.messages), 3)

        # Verify field-level evidence citations are preserved
        self.assertIn("m1", stack.terms.field_evidence.get("instrument", []))
        self.assertIn("m2", stack.terms.field_evidence.get("quantity", []))
        self.assertIn("m3", stack.terms.field_evidence.get("yield_value", []))

    # ----------------------------------------------------------------------
    # 6. Keeping price counteroffers in the same stack
    # ----------------------------------------------------------------------
    def test_price_counteroffer_in_same_stack(self):
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:test",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })
        # Counteroffer from Dealer B
        res2 = self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:test",
            "sender": "Dealer B",
            "message_text": "7.24?",
            "timestamp": "2026-11-03T10:01:00",
        })
        self.assertEqual(res2.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Counter-counter from Dealer A
        res3 = self.manager.process_message({
            "message_id": "m3",
            "chat_key": "CHAT:test",
            "sender": "Dealer A",
            "message_text": "Can do @ 7.245",
            "timestamp": "2026-11-03T10:02:00",
        })
        self.assertEqual(res3.decision, AssignmentDecision.ASSIGN_EXISTING)

        active = self.manager.get_active_stacks("CHAT:test")
        self.assertEqual(len(active), 1)
        self.assertEqual(len(active[0].messages), 3)
        self.assertEqual(active[0].terms.yield_value, 7.245)

    # ----------------------------------------------------------------------
    # 7. Handling ambiguous messages
    # ----------------------------------------------------------------------
    def test_handling_ambiguous_messages(self):
        # Two active stacks
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:multi",
            "sender": "A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:multi",
            "sender": "C",
            "message_text": "Looking to buy 2Cr GS 2027 @ 6.80",
            "timestamp": "2026-11-03T10:02:00",
        })

        # An ambiguous bare "done" with no reply-to link from an unlinked participant
        res_ambig = self.manager.process_message({
            "message_id": "m-ambig",
            "chat_key": "CHAT:multi",
            "sender": "Unknown User",
            "message_text": "Done",
            "timestamp": "2026-11-03T10:05:00",
        })
        self.assertEqual(res_ambig.decision, AssignmentDecision.UNRESOLVED)
        self.assertTrue(res_ambig.requires_human_review)

        unresolved = self.manager.get_unresolved_messages("CHAT:multi")
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["message_id"], "m-ambig")

    # ----------------------------------------------------------------------
    # 8. Excluding non-financial noise
    # ----------------------------------------------------------------------
    def test_excluding_non_financial_noise(self):
        res = self.manager.process_message({
            "message_id": "m-noise",
            "chat_key": "CHAT:general",
            "sender": "User 1",
            "message_text": "Good morning everyone!",
            "timestamp": "2026-11-03T09:00:00",
        })
        self.assertEqual(res.decision, AssignmentDecision.NOISE_EXCLUDED)
        active = self.manager.get_active_stacks("CHAT:general")
        self.assertEqual(len(active), 0)

        noise = self.manager.get_noise_messages("CHAT:general")
        self.assertEqual(len(noise), 1)
        self.assertEqual(noise[0]["message_id"], "m-noise")

    # ----------------------------------------------------------------------
    # 9. Explicit conclusion
    # ----------------------------------------------------------------------
    def test_explicit_conclusion_seals_stack(self):
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:trade",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr 7.18 SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:trade",
            "sender": "Dealer B",
            "message_text": "7.245?",
            "timestamp": "2026-11-03T10:01:00",
        })
        # Explicit conclusion
        res3 = self.manager.process_message({
            "message_id": "m3",
            "chat_key": "CHAT:trade",
            "sender": "Dealer A",
            "message_text": "done at 7.245",
            "timestamp": "2026-11-03T10:02:00",
        })

        self.assertEqual(res3.decision, AssignmentDecision.ASSIGN_EXISTING)

        active = self.manager.get_active_stacks("CHAT:trade")
        self.assertEqual(len(active), 0)  # Concluded stack is no longer active

        sealed = self.manager.get_sealed_stacks("CHAT:trade")
        self.assertEqual(len(sealed), 1)
        stack = sealed[0]
        self.assertEqual(stack.status, StackStatus.CONCLUDED)
        self.assertEqual(stack.stack_state, StackState.SEALED)
        self.assertEqual(stack.seal_reason, SealReason.EXPLICIT_CONCLUSION)
        self.assertIsNotNone(stack.sealed_at)

        events = [e.event_type for e in stack.events]
        self.assertIn("DEAL_CONFIRMED", events)

    # ----------------------------------------------------------------------
    # 10. Explicit rejection
    # ----------------------------------------------------------------------
    def test_explicit_rejection_seals_stack(self):
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:reject-test",
            "sender": "Dealer A",
            "message_text": "Selling 10Cr 6.79 GS 2027 @ 6.80",
            "timestamp": "2026-11-03T10:00:00",
        })
        # Explicit rejection
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:reject-test",
            "sender": "Dealer B",
            "message_text": "Pass, no requirement",
            "timestamp": "2026-11-03T10:02:00",
        })

        active = self.manager.get_active_stacks("CHAT:reject-test")
        self.assertEqual(len(active), 0)

        sealed = self.manager.get_sealed_stacks("CHAT:reject-test")
        self.assertEqual(len(sealed), 1)
        stack = sealed[0]
        self.assertEqual(stack.status, StackStatus.REJECTED)
        self.assertEqual(stack.stack_state, StackState.SEALED)
        self.assertEqual(stack.seal_reason, SealReason.EXPLICIT_REJECTION)

        events = [e.event_type for e in stack.events]
        self.assertIn("TRADE_REJECTED", events)

    # ----------------------------------------------------------------------
    # 11. Batch-end snapshots
    # ----------------------------------------------------------------------
    def test_batch_end_snapshot_preserves_active_pending_stacks(self):
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:batch-test",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })

        # Batch finishes: snapshot taken
        snapshot = self.manager.snapshot_batch_end("CHAT:batch-test")
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].status, StackStatus.PENDING)
        self.assertEqual(snapshot[0].stack_state, StackState.ACTIVE)

        # In-memory active stacks are still preserved as PENDING & ACTIVE
        active = self.manager.get_active_stacks("CHAT:batch-test")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].status, StackStatus.PENDING)
        self.assertEqual(active[0].stack_state, StackState.ACTIVE)

    # ----------------------------------------------------------------------
    # 12. Continuing pending negotiations across uploads
    # ----------------------------------------------------------------------
    def test_continuing_pending_negotiations_across_uploads(self):
        # Upload 1: Dealer A offers SDL 2028
        self.manager.process_message({
            "message_id": "up1-m1",
            "chat_key": "CHAT:continuing",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })
        snap1 = self.manager.snapshot_batch_end("CHAT:continuing")
        self.assertEqual(snap1[0].status, StackStatus.PENDING)

        # Upload 2: (Hours later or next batch) Dealer B responds to that exact negotiation
        res2 = self.manager.process_message({
            "message_id": "up2-m2",
            "chat_key": "CHAT:continuing",
            "sender": "Dealer B",
            "message_text": "Can do 5Cr SDL 2028 @ 7.24 done",
            "timestamp": "2026-11-03T10:30:00",
        })
        self.assertEqual(res2.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Should be the same stack, now concluded
        sealed = self.manager.get_sealed_stacks("CHAT:continuing")
        self.assertEqual(len(sealed), 1)
        self.assertEqual(sealed[0].status, StackStatus.CONCLUDED)
        self.assertEqual(len(sealed[0].messages), 2)

    # ----------------------------------------------------------------------
    # 13. Preventing duplicate message assignments
    # ----------------------------------------------------------------------
    def test_preventing_duplicate_message_assignments(self):
        msg = {
            "message_id": "msg-dup-1",
            "chat_key": "CHAT:dup-test",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        }
        res1 = self.manager.process_message(msg)
        # Re-processing exact same message
        res2 = self.manager.process_message(msg)

        self.assertEqual(res1.assigned_stack_id, res2.assigned_stack_id)
        active = self.manager.get_active_stacks("CHAT:dup-test")
        self.assertEqual(len(active), 1)
        # Must only have 1 message in stack, not duplicated!
        self.assertEqual(len(active[0].messages), 1)

    # ----------------------------------------------------------------------
    # 14. Handling late-arriving messages without altering concluded deal
    # ----------------------------------------------------------------------
    def test_late_arriving_messages_on_concluded_stack(self):
        # 1. Message 1 @ 10:00
        self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:late-test",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr 7.18 SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })
        # 2. Concluded @ 10:02
        self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:late-test",
            "sender": "Dealer B",
            "message_text": "done at 7.25",
            "timestamp": "2026-11-03T10:02:00",
        })
        sealed = self.manager.get_sealed_stacks("CHAT:late-test")
        self.assertEqual(len(sealed), 1)
        self.assertEqual(sealed[0].status, StackStatus.CONCLUDED)

        # 3. Late-arriving message from 10:01 attempting counteroffer arrives later
        late_msg = {
            "message_id": "m-late",
            "chat_key": "CHAT:late-test",
            "sender": "Dealer B",
            "message_text": "Counter @ 7.24",
            "reply_to_message_id": "m1",
            "timestamp": "2026-11-03T10:01:00",  # earlier timestamp!
        }
        res_late = self.manager.process_message(late_msg)
        self.assertEqual(res_late.decision, AssignmentDecision.ASSIGN_EXISTING)

        # Terminal status CONCLUDED must remain protected!
        sealed = self.manager.get_sealed_stacks("CHAT:late-test")
        self.assertEqual(sealed[0].status, StackStatus.CONCLUDED)
        self.assertEqual(sealed[0].stack_state, StackState.SEALED)
        # Audit event recorded for message appended to sealed stack
        events = [e.event_type for e in sealed[0].events]
        self.assertIn("MESSAGE_APPENDED_TO_SEALED_STACK", events)

    # ----------------------------------------------------------------------
    # 15. Ensuring one chat cannot contaminate another chat's stacks
    # ----------------------------------------------------------------------
    def test_chat_isolation(self):
        # Chat 1: SDL 2028
        self.manager.process_message({
            "message_id": "c1-m1",
            "chat_key": "CHAT:group-1",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr SDL 2028 @ 7.25",
            "timestamp": "2026-11-03T10:00:00",
        })

        # Chat 2: Inquires "price?" (should NOT match Chat 1's stack!)
        res_c2 = self.manager.process_message({
            "message_id": "c2-m1",
            "chat_key": "CHAT:group-2",
            "sender": "Dealer B",
            "message_text": "What is the price?",
            "timestamp": "2026-11-03T10:01:00",
        })

        # In Chat 2, there are no active stacks, so "price?" without context is unresolved/noise,
        # and NEVER assigned to Chat 1's stack!
        self.assertNotEqual(res_c2.decision, AssignmentDecision.ASSIGN_EXISTING)
        chat1_stacks = self.manager.get_active_stacks("CHAT:group-1")
        chat2_stacks = self.manager.get_active_stacks("CHAT:group-2")

        self.assertEqual(len(chat1_stacks), 1)
        self.assertEqual(len(chat2_stacks), 0)
        self.assertEqual(len(chat1_stacks[0].messages), 1)


if __name__ == "__main__":
    unittest.main()

