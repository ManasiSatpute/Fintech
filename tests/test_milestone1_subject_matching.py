"""
Unit tests for Milestone 1: Bond Subject Identification and Message Matching.

Verifies:
1. Credit rating extraction and matching (AAA, AA+, AA-, SOV, CRISIL/ICRA/CARE/IND).
2. Credit rating conflict detection (AAA vs AA-).
3. Industry / sector identification and mapping (NBFC, PSU, HFC, Power, Banking).
4. Industry conflict detection (PSU vs NBFC).
5. Tenor extraction ("5y", "10 year", "3 yr tenor", "2.5y").
6. Tenor conflict detection ("10Y" vs "3Y").
7. Spread extraction and matching ("+25 bps", "spread 35bps").
8. Deterministic subject fingerprint generation.
9. Multiple simultaneous negotiations separated by rating, tenor, or industry.
"""

import unittest

from deal_stack_manager import DealStackManager
from deal_stack_matcher import (
    _check_hard_conflicts,
    compute_subject_fingerprint,
    extract_bond_features,
    match_message_to_stacks,
    score_stack_candidate,
)
from deal_stack_models import (
    AssignmentDecision,
    DealStack,
    DealTerms,
    EvidenceTier,
    MessageBondFeatures,
    StackStatus,
)


class Milestone1SubjectMatchingTests(unittest.TestCase):

    def setUp(self):
        self.manager = DealStackManager()

    # ----------------------------------------------------------------------
    # 1. Credit Rating Extraction & Matching
    # ----------------------------------------------------------------------
    def test_credit_rating_extraction(self):
        # CRISIL AAA
        f1 = extract_bond_features({
            "message_text": "Selling 10Cr CRISIL AAA 2028 @ 7.25",
            "message_id": "m1",
        })
        self.assertEqual(f1.credit_rating, "AAA")
        self.assertTrue(f1.has_bond_terms)

        # ICRA AA+
        f2 = extract_bond_features({
            "message_text": "Looking for 5Cr ICRA AA+ paper",
            "message_id": "m2",
        })
        self.assertEqual(f2.credit_rating, "AA+")

        # Sovereign / SOV
        f3 = extract_bond_features({
            "message_text": "Buying 25Cr SOV 7.18 GS 2033",
            "message_id": "m3",
        })
        self.assertEqual(f3.credit_rating, "SOV")

        # Rated AA-
        f4 = extract_bond_features({
            "message_text": "Oxyzo rated AA- 50Cr at 8.75",
            "message_id": "m4",
        })
        self.assertEqual(f4.credit_rating, "AA-")

    def test_credit_rating_match_score(self):
        feat = extract_bond_features({
            "message_text": "Can do 10Cr AAA @ 7.24",
            "message_id": "m-reply",
            "sender": "Dealer B",
        })
        stack = DealStack(
            chat_key="CHAT:test",
            credit_rating="AAA",
            opened_at="2026-11-13T10:00:00",
            last_activity_at="2026-11-13T10:00:00",
        )
        score = score_stack_candidate(feat, stack)
        self.assertFalse(score.is_conflicting_bond)
        self.assertIn(EvidenceTier.MATURITY_COUPON_RATING.value, score.breakdown)
        self.assertTrue(any("Credit rating match" in r for r in score.reasons))

    def test_credit_rating_conflict_triggers_new_stack(self):
        feat = extract_bond_features({
            "message_text": "Looking to buy 10Cr AA- paper @ 8.20",
            "message_id": "m-conflict",
            "sender": "Dealer C",
        })
        stack = DealStack(
            chat_key="CHAT:test",
            issuer="PFC",
            credit_rating="AAA",
            opened_at="2026-11-13T10:00:00",
            last_activity_at="2026-11-13T10:00:00",
        )
        is_conflict, reason = _check_hard_conflicts(feat, stack)
        self.assertTrue(is_conflict)
        self.assertIn("credit rating", reason.lower())

        score = score_stack_candidate(feat, stack)
        self.assertTrue(score.is_conflicting_bond)
        self.assertEqual(score.total_score, 0.0)

    # ----------------------------------------------------------------------
    # 2. Industry / Sector Identification & Matching
    # ----------------------------------------------------------------------
    def test_industry_identification_from_issuer(self):
        f_pfc = extract_bond_features({"message_text": "Selling 10Cr PFC 2028"})
        self.assertEqual(f_pfc.industry, "PSU")

        f_muthoot = extract_bond_features({"message_text": "Selling 5Cr Muthoot 2026"})
        self.assertEqual(f_muthoot.industry, "NBFC")

        f_lichfl = extract_bond_features({"message_text": "Buying 15Cr LIC Housing 2027"})
        self.assertEqual(f_lichfl.industry, "HOUSING_FINANCE")

        f_sbi = extract_bond_features({"message_text": "SBI Tier 2 2030 @ 7.45"})
        self.assertEqual(f_sbi.industry, "BANKING")

        f_meghalaya = extract_bond_features({"message_text": "Meghalaya Energy 2028"})
        self.assertEqual(f_meghalaya.industry, "POWER")

    def test_industry_identification_from_explicit_keywords(self):
        f_nbfc = extract_bond_features({"message_text": "Looking for AAA NBFC paper 5y"})
        self.assertEqual(f_nbfc.industry, "NBFC")

        f_psu = extract_bond_features({"message_text": "Have 20Cr PSU bond available"})
        self.assertEqual(f_psu.industry, "PSU")

    def test_industry_match_and_conflict(self):
        # Match
        feat_psu = extract_bond_features({"message_text": "Any other PSU bonds?"})
        stack_psu = DealStack(
            chat_key="CHAT:test",
            industry="PSU",
            opened_at="2026-11-13T10:00:00",
            last_activity_at="2026-11-13T10:00:00",
        )
        score = score_stack_candidate(feat_psu, stack_psu)
        self.assertFalse(score.is_conflicting_bond)
        self.assertTrue(any("Industry sector match" in r for r in score.reasons))

        # Conflict: Explicit NBFC vs PSU stack
        feat_nbfc = extract_bond_features({"message_text": "Selling 10Cr NBFC paper"})
        is_conflict, reason = _check_hard_conflicts(feat_nbfc, stack_psu)
        self.assertTrue(is_conflict)
        self.assertIn("industry", reason.lower())

    # ----------------------------------------------------------------------
    # 3. Tenor Extraction & Matching
    # ----------------------------------------------------------------------
    def test_tenor_extraction(self):
        f1 = extract_bond_features({"message_text": "Selling 5Cr 5y SDL 2028"})
        self.assertEqual(f1.tenor, "5Y")
        self.assertEqual(f1.maturity, "2028")

        f2 = extract_bond_features({"message_text": "Looking for 10 year PSU paper"})
        self.assertEqual(f2.tenor, "10Y")

        f3 = extract_bond_features({"message_text": "3 yr tenor @ 7.30"})
        self.assertEqual(f3.tenor, "3Y")

        f4 = extract_bond_features({"message_text": "2.5y paper available"})
        self.assertEqual(f4.tenor, "2.5Y")

    def test_tenor_conflict(self):
        feat = extract_bond_features({"message_text": "Need 3y paper"})
        stack = DealStack(
            chat_key="CHAT:test",
            tenor="10Y",
            opened_at="2026-11-13T10:00:00",
            last_activity_at="2026-11-13T10:00:00",
        )
        is_conflict, reason = _check_hard_conflicts(feat, stack)
        self.assertTrue(is_conflict)
        self.assertIn("tenor", reason.lower())

    # ----------------------------------------------------------------------
    # 4. Spread Extraction & Matching
    # ----------------------------------------------------------------------
    def test_spread_extraction(self):
        f1 = extract_bond_features({"message_text": "Offering at +25 bps over G-sec"})
        self.assertEqual(f1.spread, 25.0)
        self.assertEqual(f1.spread_raw, "+25 bps")

        f2 = extract_bond_features({"message_text": "Spread 35bps"})
        self.assertEqual(f2.spread, 35.0)

    # ----------------------------------------------------------------------
    # 5. Deterministic Fingerprint
    # ----------------------------------------------------------------------
    def test_subject_fingerprint_generation(self):
        terms = DealTerms(
            issuer="PFC",
            instrument="NCD",
            industry="PSU",
            maturity="2028",
            tenor="5Y",
            credit_rating="AAA",
            coupon_rate=7.25,
        )
        fp = compute_subject_fingerprint(terms)
        self.assertEqual(fp, "BOND:PFC:NCD:PSU:2028:5Y:AAA:7.25%")

        # Same ISIN always takes precedence
        terms_with_isin = DealTerms(isin="INE134E08KO9", issuer="PFC")
        self.assertEqual(compute_subject_fingerprint(terms_with_isin), "ISIN:INE134E08KO9")

    # ----------------------------------------------------------------------
    # 6. Simultaneous Negotiations Separated by Rating / Tenor / Industry
    # ----------------------------------------------------------------------
    def test_simultaneous_negotiations_separated_by_rating(self):
        # 10:00 - Dealer A: Selling 10Cr PFC AAA 2028 @ 7.25
        res1 = self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:group-multi",
            "sender": "Dealer A",
            "message_text": "Selling 10Cr PFC AAA 2028 @ 7.25",
            "timestamp": "2026-11-13T10:00:00",
        })
        self.assertEqual(res1.decision, AssignmentDecision.NEW_STACK)

        # 10:01 - Dealer C: Looking for 10Cr PFC AA- 2028 @ 8.10 (Different rating tranche!)
        res2 = self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:group-multi",
            "sender": "Dealer C",
            "message_text": "Looking for 10Cr PFC AA- 2028 @ 8.10",
            "timestamp": "2026-11-13T10:01:00",
        })
        self.assertEqual(res2.decision, AssignmentDecision.NEW_STACK)

        active = self.manager.get_active_stacks("CHAT:group-multi")
        self.assertEqual(len(active), 2)
        stack_aaa = next(s for s in active if s.credit_rating == "AAA")
        stack_aa_minus = next(s for s in active if s.credit_rating == "AA-")
        self.assertNotEqual(stack_aaa.stack_id, stack_aa_minus.stack_id)

    def test_simultaneous_negotiations_separated_by_tenor(self):
        # Dealer A: 5y paper
        res1 = self.manager.process_message({
            "message_id": "m1",
            "chat_key": "CHAT:tenors",
            "sender": "Dealer A",
            "message_text": "Selling 5Cr 5y SDL 2028 @ 7.25",
            "timestamp": "2026-11-13T10:00:00",
        })
        self.assertEqual(res1.decision, AssignmentDecision.NEW_STACK)

        # Dealer B: 10y paper
        res2 = self.manager.process_message({
            "message_id": "m2",
            "chat_key": "CHAT:tenors",
            "sender": "Dealer B",
            "message_text": "Selling 10Cr 10y SDL 2033 @ 7.40",
            "timestamp": "2026-11-13T10:01:00",
        })
        self.assertEqual(res2.decision, AssignmentDecision.NEW_STACK)

        active = self.manager.get_active_stacks("CHAT:tenors")
        self.assertEqual(len(active), 2)


if __name__ == "__main__":
    unittest.main()

