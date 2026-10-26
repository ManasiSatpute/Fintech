import unittest
from datetime import datetime
from unittest.mock import patch

from scripts.baseline_report import pipeline


SNIPPET = """Chat: Synthetic Desk
Type: GROUP
Date: 2026-10-20
#1 [10:00:00] [OWN] Desk: Holding bid at 10.08 for 5 cr Synthetic GS Aug 2030 T+1
#2 [10:02:00] Trader B: closed
#3 [10:03:00] Trader B: kal T+1
"""


def concluded_deal():
    return {
        "bond_identity_text": "Synthetic GS Aug 2030",
        "isin_text": None,
        "issuer_text": None,
        "maturity_text": "Aug 2030",
        "coupon_rate": None,
        "credit_rating": None,
        "quantity_text": "5 cr",
        "ytm": "10.08",
        "ytm_is_two_way": False,
        "price": None,
        "settlement_text": "T+1",
        "owner_side": "BUY",
        "buyer_text": "Desk",
        "seller_text": "Trader B",
        "rfq_id": None,
        "package_note": None,
        "is_indicative": False,
        "status": "CONCLUDED",
        "status_evidence": "closed",
        "discard_reason": None,
        "evidence": {
            "bond_identity_text": ["#1"],
            "maturity_text": ["#1"],
            "quantity_text": ["#1"],
            "ytm": ["#1"],
            "settlement_text": ["#1"],
            "owner_side": ["#1"],
            "buyer_text": ["#1"],
            "seller_text": ["#2"],
            "is_indicative": ["#1"],
            "status": ["#2"],
            "status_evidence": ["#2"],
        },
    }


class EvidenceValidationTests(unittest.TestCase):
    def validate_direction(self, message, owner_side, buyer_text, seller_text):
        deal = {
            "owner_side": owner_side,
            "buyer_text": buyer_text,
            "seller_text": seller_text,
            "evidence": {
                "owner_side": ["#1"],
                "buyer_text": ["#1"] if buyer_text else [],
                "seller_text": ["#1"] if seller_text else [],
            },
        }
        return pipeline.validate_extraction(
            deal,
            (
                "Chat: Synthetic Desk\n"
                "Type: PRIVATE\n"
                "Date: 2026-10-20\n"
                f"#1 [10:00:00] [OWN] Desk: {message}"
            ),
        )

    def test_offer_requests_mean_sender_wants_to_buy(self):
        for phrase in (
            "offer pls",
            "offer please",
            "need offer",
            "any offer?",
            "offers?",
        ):
            with self.subTest(phrase=phrase):
                result = self.validate_direction(
                    phrase,
                    "BUY",
                    "Desk",
                    None,
                )
                self.assertEqual(result["owner_side"], "BUY")
                self.assertEqual(result["buyer_text"], "Desk")
                self.assertIsNone(result["seller_text"])

    def test_offer_request_corrects_an_inconsistent_model_side(self):
        result = self.validate_direction(
            "offer pls",
            "SELL",
            "Desk",
            None,
        )

        self.assertEqual(result["owner_side"], "BUY")
        self.assertEqual(result["buyer_text"], "Desk")
        self.assertIn("side_corrected", result["warnings"])
        self.assertNotIn("unsupported_owner_side", result["warnings"])

    def test_level_bearing_offers_mean_sender_is_selling(self):
        for phrase in (
            "offer at 10.08",
            "I have an offer",
            "10% offer",
        ):
            with self.subTest(phrase=phrase):
                result = self.validate_direction(
                    phrase,
                    "SELL",
                    None,
                    "Desk",
                )
                self.assertEqual(result["owner_side"], "SELL")
                self.assertIsNone(result["buyer_text"])
                self.assertEqual(result["seller_text"], "Desk")

    def test_bid_requests_mean_sender_wants_to_sell(self):
        for phrase in ("bid pls", "bid please", "any bids?"):
            with self.subTest(phrase=phrase):
                result = self.validate_direction(
                    phrase,
                    "SELL",
                    None,
                    "Desk",
                )
                self.assertEqual(result["owner_side"], "SELL")
                self.assertIsNone(result["buyer_text"])
                self.assertEqual(result["seller_text"], "Desk")

    def test_bid_request_corrects_an_inconsistent_model_side(self):
        result = self.validate_direction(
            "bid pls",
            "BUY",
            None,
            None,
        )

        self.assertEqual(result["owner_side"], "SELL")
        self.assertIn("side_corrected", result["warnings"])
        self.assertNotIn("unsupported_owner_side", result["warnings"])

    def test_level_bearing_bids_mean_sender_is_buying(self):
        for phrase in ("bid at 10.08", "I have a bid"):
            with self.subTest(phrase=phrase):
                result = self.validate_direction(
                    phrase,
                    "BUY",
                    "Desk",
                    None,
                )
                self.assertEqual(result["owner_side"], "BUY")
                self.assertEqual(result["buyer_text"], "Desk")
                self.assertIsNone(result["seller_text"])

    def test_isin_units_settlement_without_side_keeps_parties_empty(self):
        result = pipeline.validate_extraction(
            {
                "isin_text": "ZZ0000000000",
                "quantity_text": "43 units",
                "settlement_text": "T+1",
                "owner_side": "UNKNOWN",
                "buyer_text": None,
                "seller_text": None,
                "evidence": {
                    "isin_text": ["#1"],
                    "quantity_text": ["#1"],
                    "settlement_text": ["#1"],
                    "owner_side": ["#1"],
                },
            },
            (
                "Chat: Synthetic Desk\n"
                "Type: PRIVATE\n"
                "Date: 2026-10-20\n"
                "#1 [10:00:00] [OWN] Desk: "
                "ZZ0000000000 43 units T+1"
            ),
            datetime(2026, 10, 20, 10, 0),
        )

        self.assertEqual(result["owner_side"], "UNKNOWN")
        self.assertIsNone(result["buyer_text"])
        self.assertIsNone(result["seller_text"])

    def test_isin_units_without_side_clears_model_party_values(self):
        result = pipeline.validate_extraction(
            {
                "isin_text": "US0378331005",
                "quantity_text": "158 units",
                "owner_side": "BUY",
                "buyer_text": "Desk",
                "seller_text": "Trader B",
                "evidence": {
                    "isin_text": ["#1"],
                    "quantity_text": ["#1"],
                    "owner_side": ["#1"],
                    "buyer_text": ["#1"],
                    "seller_text": ["#1"],
                },
            },
            (
                "Chat: Synthetic Desk\n"
                "Type: PRIVATE\n"
                "Date: 2026-10-20\n"
                "#1 [10:00:00] [OWN] Desk: "
                "US0378331005 158 units"
            ),
        )

        self.assertEqual(result["owner_side"], "UNKNOWN")
        self.assertIsNone(result["buyer_text"])
        self.assertIsNone(result["seller_text"])
        self.assertIn("side_corrected", result["warnings"])

    def test_missing_status_defaults_to_uncertain_with_warning(self):
        deal = concluded_deal()
        deal.pop("status")
        deal["evidence"].pop("status")

        result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIn("status_missing_defaulted", result["warnings"])

    def test_unrecognized_status_defaults_to_uncertain_with_warning(self):
        deal = concluded_deal()
        deal["status"] = "NOT_A_STATUS"

        result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIn("status_missing_defaulted", result["warnings"])

    def test_valid_status_is_preserved_without_default_warning(self):
        deal = concluded_deal()

        result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertEqual(result["status"], "CONCLUDED")
        self.assertNotIn("status_missing_defaulted", result["warnings"])

    def test_status_rejected_by_evidence_defaults_to_uncertain(self):
        deal = concluded_deal()
        deal["evidence"]["status"] = ["#1"]

        result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIn("status_missing_defaulted", result["warnings"])

    def test_valid_conclusion_has_normalized_fields_and_verified_confidence(self):
        with patch.object(pipeline.config, "REVIEW_CONFIDENCE_THRESHOLD", 0.7):
            result = pipeline.validate_extraction(concluded_deal(), SNIPPET)

        self.assertEqual(result["status"], "CONCLUDED")
        self.assertEqual(result["normalized_quantity"]["amount_inr"], 50_000_000)
        self.assertEqual(result["ytm_values"], [10.08])
        self.assertEqual(result["settlement_date"], "2026-10-21")
        self.assertEqual(result["settlement_label"], "T1")
        self.assertEqual(result["maturity_ym"], "2030-08")
        self.assertIsNone(result["maturity_date"])
        self.assertEqual(result["confidence"], 1.0)
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["missing_fields"], [])
        self.assertEqual(result["errors"], [])
        self.assertNotIn("final_price", result)

    def test_unsupported_numeric_value_is_null_with_warning_and_downgrade(self):
        deal = concluded_deal()
        deal["ytm"] = "11.00"
        with patch.object(pipeline.config, "REVIEW_CONFIDENCE_THRESHOLD", 0.9):
            result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertIsNone(result["ytm"])
        self.assertIn("unsupported_ytm", result["warnings"])
        self.assertIn("ytm", result["missing_fields"])
        self.assertEqual(result["status"], "NEGOTIATING")
        self.assertTrue(result["needs_review"])

    def test_isin_must_match_verbatim_in_its_evidence_lines(self):
        deal = concluded_deal()
        deal["isin_text"] = "US0378331005"
        deal["evidence"]["isin_text"] = ["#2"]
        result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertIsNone(result["isin_text"])
        self.assertIn("unsupported_isin_text", result["warnings"])

    def test_line_number_and_timestamp_digits_do_not_support_a_numeric_value(self):
        deal = concluded_deal()
        deal["ytm"] = "1"
        deal["evidence"]["ytm"] = ["#1"]
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertIsNone(result["ytm"])
        self.assertIn("unsupported_ytm", result["warnings"])

    def test_conclusion_requires_later_acceptance_and_key_terms(self):
        deal = concluded_deal()
        deal["evidence"]["status_evidence"] = ["#1"]
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertEqual(result["status"], "NEGOTIATING")
        self.assertIn("later_acceptance_evidence", result["missing_fields"])

        unrelated_ack = concluded_deal()
        unrelated_ack["status_evidence"] = "ok"
        unrelated_ack["evidence"]["status_evidence"] = ["#3"]
        result = pipeline.validate_extraction(unrelated_ack, SNIPPET)
        self.assertNotEqual(result["status"], "CONCLUDED")
        self.assertIn("later_acceptance_evidence", result["missing_fields"])

        missing_terms = concluded_deal()
        missing_terms["quantity_text"] = None
        result = pipeline.validate_extraction(missing_terms, SNIPPET)
        self.assertEqual(result["status"], "NEGOTIATING")
        self.assertIn("quantity_text", result["missing_fields"])

    def test_cancellation_requires_refusal_as_next_message_after_offer(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: PRIVATE\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering Synthetic GS Aug 2030 at 10.08 for 5 cr\n"
            "#2 [10:01:00] Trader B: no requirement\n"
        )
        deal = concluded_deal()
        deal["status"] = "CANCELLED"
        deal["status_evidence"] = "no requirement"
        deal["evidence"]["status"] = ["#2"]
        deal["evidence"]["status_evidence"] = ["#2"]

        result = pipeline.validate_extraction(deal, snippet)

        self.assertEqual(result["status"], "CANCELLED")
        self.assertEqual(result["status_evidence"], "no requirement")
        self.assertNotIn("linked_offer_or_inquiry", result["missing_fields"])
        self.assertNotIn("unlinked_cancellation_refusal", result["warnings"])

    def test_cancellation_can_link_to_offer_through_reply_quote(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: PRIVATE\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offer Synthetic GS Aug 2030 at 10.08 for 5 cr\n"
            "#2 [10:01:00] Trader B: Checking\n"
            "#3 [10:02:00] Trader B: nahi\n"
            "    replying to: Offer Synthetic GS Aug 2030 at 10.08 for 5 cr\n"
        )
        deal = concluded_deal()
        deal["status"] = "CANCELLED"
        deal["status_evidence"] = "nahi"
        deal["evidence"]["status"] = ["#3"]
        deal["evidence"]["status_evidence"] = ["#3"]

        result = pipeline.validate_extraction(deal, snippet)

        self.assertEqual(result["status"], "CANCELLED")
        self.assertEqual(result["status_evidence"], "nahi")
        self.assertNotIn("linked_offer_or_inquiry", result["missing_fields"])

    def test_unlinked_refusal_is_uncertain_and_silence_stays_initiated(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: PRIVATE\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] Trader B: Checking\n"
            "#2 [10:01:00] Trader B: rehne do\n"
            "#3 [10:02:00] [OWN] Desk: Offering Synthetic GS Aug 2030 at 10.08 for 5 cr\n"
        )
        deal = concluded_deal()
        deal["status"] = "CANCELLED"
        deal["status_evidence"] = "rehne do"
        deal["evidence"]["status"] = ["#2"]
        deal["evidence"]["status_evidence"] = ["#2"]

        result = pipeline.validate_extraction(deal, snippet)
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIn("linked_offer_or_inquiry", result["missing_fields"])
        self.assertIn("unlinked_cancellation_refusal", result["warnings"])

        silent_deal = concluded_deal()
        silent_deal["status"] = "INITIATED"
        silent_result = pipeline.validate_extraction(silent_deal, snippet)
        self.assertEqual(silent_result["status"], "INITIATED")

    def test_buyer_seller_equality_only_errors_when_both_are_known(self):
        deal = concluded_deal()
        deal["seller_text"] = "Desk"
        deal["evidence"]["seller_text"] = ["#1"]
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertIn("buyer_equals_seller", result["errors"])

        deal["seller_text"] = None
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertNotIn("buyer_equals_seller", result["errors"])

    def test_normalizer_review_flag_is_propagated_and_threshold_is_applied(self):
        deal = concluded_deal()
        deal["settlement_text"] = "kal T+1"
        deal["evidence"]["settlement_text"] = ["#3"]
        with patch.object(pipeline.config, "REVIEW_CONFIDENCE_THRESHOLD", 1.0):
            result = pipeline.validate_extraction(deal, SNIPPET)

        self.assertTrue(result["settlement_needs_review"])
        self.assertTrue(result["needs_review"])
        self.assertIn("settlement_needs_review", result["warnings"])

    def test_ytm_normalised_to_plain_number_without_percent_sign(self):
        deal = concluded_deal()
        deal["ytm"] = "10.08%"
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertEqual(result["ytm"], 10.08)
        self.assertIsInstance(result["ytm"], float)
        self.assertEqual(result["ytm_values"], [10.08])
        self.assertNotIn("yield_out_of_expected_range", result["warnings"])

    def test_yield_outside_range_high_is_null_with_warning_and_needs_review(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: GROUP\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering Synthetic GS Aug 2030 at 27.50% for 5 cr\n"
            "#2 [10:02:00] Trader B: closed\n"
        )
        deal = concluded_deal()
        deal["ytm"] = "27.50%"
        result = pipeline.validate_extraction(deal, snippet)
        self.assertIsNone(result["ytm"])
        self.assertIn("yield_out_of_expected_range", result["warnings"])
        self.assertTrue(result["needs_review"])
        self.assertIn("ytm", result["missing_fields"])

    def test_yield_outside_range_low_is_null_with_warning_and_needs_review(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: GROUP\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering Synthetic GS Aug 2030 at 2.50% for 5 cr\n"
            "#2 [10:02:00] Trader B: closed\n"
        )
        deal = concluded_deal()
        deal["ytm"] = "2.50%"
        result = pipeline.validate_extraction(deal, snippet)
        self.assertIsNone(result["ytm"])
        self.assertIn("yield_out_of_expected_range", result["warnings"])
        self.assertTrue(result["needs_review"])
        self.assertIn("ytm", result["missing_fields"])

    def test_yield_boundary_values_are_valid(self):
        snippet_3 = (
            "Chat: Synthetic Desk\n"
            "Type: GROUP\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering Synthetic GS Aug 2030 at 3.00% for 5 cr\n"
            "#2 [10:02:00] Trader B: closed\n"
        )
        deal_3 = concluded_deal()
        deal_3["ytm"] = "3.00%"
        result_3 = pipeline.validate_extraction(deal_3, snippet_3)
        self.assertEqual(result_3["ytm"], 3.0)
        self.assertNotIn("yield_out_of_expected_range", result_3["warnings"])

        snippet_25 = (
            "Chat: Synthetic Desk\n"
            "Type: GROUP\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering Synthetic GS Aug 2030 at 25.00% for 5 cr\n"
            "#2 [10:02:00] Trader B: closed\n"
        )
        deal_25 = concluded_deal()
        deal_25["ytm"] = "25.00%"
        result_25 = pipeline.validate_extraction(deal_25, snippet_25)
        self.assertEqual(result_25["ytm"], 25.0)
        self.assertNotIn("yield_out_of_expected_range", result_25["warnings"])

    def test_partial_isin_fragment_moves_to_partial_isin_text_and_sets_isin_null(self):
        deal = concluded_deal()
        deal["isin_text"] = "12345"
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertIsNone(result["isin_text"])
        self.assertEqual(result["partial_isin_text"], "12345")
        self.assertIn("partial_isin", result["warnings"])

    def test_invalid_checksum_isin_moves_to_partial_isin_text_and_sets_isin_null(self):
        deal = concluded_deal()
        deal["isin_text"] = "US0378331006"
        result = pipeline.validate_extraction(deal, SNIPPET)
        self.assertIsNone(result["isin_text"])
        self.assertEqual(result["partial_isin_text"], "US0378331006")
        self.assertIn("partial_isin", result["warnings"])

    def test_full_valid_isin_retains_isin_text_without_partial_isin_warning(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: GROUP\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering 5 cr US0378331005 at 10.08\n"
            "#2 [10:02:00] Trader B: closed\n"
        )
        deal = concluded_deal()
        deal["isin_text"] = "US0378331005"
        deal["evidence"]["isin_text"] = ["#1"]
        result = pipeline.validate_extraction(deal, snippet)
        self.assertEqual(result["isin_text"], "US0378331005")
        self.assertIsNone(result["partial_isin_text"])
        self.assertNotIn("partial_isin", result["warnings"])
        self.assertNotIn("unsupported_isin_text", result["warnings"])

    def test_acceptance_after_earliest_term_line_allows_concluded_status(self):
        snippet = (
            "Chat: Synthetic Desk\n"
            "Type: GROUP\n"
            "Date: 2026-10-20\n"
            "#1 [10:00:00] [OWN] Desk: Offering 5 cr Synthetic GS Aug 2030 T+1\n"
            "#2 [10:01:00] [OWN] Desk: Level is 10.08%\n"
            "#3 [10:02:00] Trader B: Checking\n"
            "#4 [10:03:00] Trader B: closed\n"
            "#5 [10:04:00] Trader B: ok\n"
            "#6 [10:05:00] [OWN] Desk: Thanks\n"
            "#7 [10:06:00] Trader B: Details sent\n"
            "#8 [10:07:00] [OWN] Desk: Noted\n"
            "#9 [10:08:00] Trader B: Confirmed at 10.08%\n"
        )
        deal = concluded_deal()
        deal["ytm"] = "10.08%"
        deal["evidence"]["ytm"] = ["#2", "#9"]
        deal["evidence"]["status"] = ["#4"]
        deal["evidence"]["status_evidence"] = ["#4"]
        deal["status_evidence"] = "closed"
        result = pipeline.validate_extraction(deal, snippet)
        self.assertEqual(result["status"], "CONCLUDED")
        self.assertNotIn("later_acceptance_evidence", result["missing_fields"])
        self.assertNotIn("conclusion_downgraded", result["warnings"])


if __name__ == "__main__":
    unittest.main()
