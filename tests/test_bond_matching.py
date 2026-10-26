import csv
import tempfile
import unittest
from pathlib import Path

from bond_matching import (
    BondFieldEvidence,
    BondMatchRequest,
    BondMatchStatus,
    match_bond,
)
from bond_reference import BondReferenceError, BondReferenceIndex
from isin_utils import (
    IsinStatus,
    LinkedMessageEvidence,
    extract_isin_from_message,
    validate_llm_isin,
)


FIELDS = (
    "ISIN",
    "NAME_OF_ISSUER",
    "SECURITY_DESCRIPTION",
    "MATURITY_DATE",
    "INTEREST_RATE",
    "CURRENCY",
)


class BondMatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "synthetic-reference.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def reference_index(self, rows=None):
        if rows is None:
            rows = [
                (
                    "US0378331005",
                    "Synthetic Issuer",
                    "Synthetic Bond 2030",
                    "14-Jan-2030",
                    "8.5000",
                    "INR",
                ),
                (
                    "US5949181045",
                    "Synthetic Other",
                    "Another Synthetic Bond",
                    "15-Feb-2031",
                    "9.2500",
                    "INR",
                ),
            ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        return BondReferenceIndex.load(self.csv_path)

    @staticmethod
    def evidence(text, value, field_name, message_id="trade-1"):
        start = text.index(value)
        return BondFieldEvidence(
            field_name=field_name,
            message_id=message_id,
            source_message_text=text,
            start=start,
            end=start + len(value),
        )

    def test_valid_source_supported_isin_matches_reference(self):
        index = self.reference_index()
        source = "Trade for us-037833-1005."
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(source, "trade-1"),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.PRIMARY_ISIN_MATCH)
        self.assertEqual(result.identity_level, "ISIN-confirmed")
        self.assertEqual(result.matching_method, "primary_isin")
        self.assertEqual(result.matched_reference.normalized_isin, "US0378331005")
        self.assertEqual(len(result.isin_source_evidence), 1)
        self.assertEqual(
            result.isin_source_evidence[0].matched_text,
            "us-037833-1005",
        )

    def test_valid_isin_absent_from_reference_is_distinguished(self):
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(
                    "Trade for US0378331005.", "trade-1"
                ),
            ),
            self.reference_index(rows=[
                (
                    "US5949181045",
                    "Synthetic Other",
                    "Another Synthetic Bond",
                    "15-Feb-2031",
                    "9.2500",
                    "INR",
                ),
            ]),
        )

        self.assertEqual(result.status, BondMatchStatus.VALID_ISIN_NOT_FOUND)
        self.assertIsNone(result.matched_reference)
        self.assertEqual(result.normalized_isin, "US0378331005")

    def test_conflicting_reference_rows_for_one_isin_are_ambiguous(self):
        index = self.reference_index(rows=[
            (
                "US0378331005",
                "Synthetic Issuer A",
                "Synthetic Security A",
                "14-Jan-2030",
                "8.5000",
                "INR",
            ),
            (
                "US0378331005",
                "Synthetic Issuer B",
                "Synthetic Security B",
                "15-Feb-2031",
                "9.2500",
                "INR",
            ),
        ])
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(
                    "US0378331005", "trade-1"
                ),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.AMBIGUOUS_MATCH)
        self.assertIsNone(result.matched_reference)
        self.assertEqual(len(result.candidate_references), 2)

    def test_multiple_source_isins_are_ambiguous_not_selected(self):
        index = self.reference_index()
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(
                    "US0378331005 or US5949181045", "trade-1"
                ),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.AMBIGUOUS_MATCH)
        self.assertIsNone(result.matched_reference)
        self.assertEqual(
            {record.normalized_isin for record in result.candidate_references},
            {"US0378331005", "US5949181045"},
        )

    def test_invalid_format_and_checksum_failure_are_never_looked_up(self):
        lookups = []

        class LookupMustNotRun:
            def lookup_isin(self, isin):
                lookups.append(("single", isin))
                return None

            def lookup_isin_candidates(self, isin):
                lookups.append(("candidates", isin))
                return ()

        for text, expected_status, phase3_status in (
            ("ISIN: 123456", BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
             IsinStatus.INVALID_FORMAT),
            ("ISIN: US0378331006", BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
             IsinStatus.CHECKSUM_FAILURE),
        ):
            with self.subTest(text=text):
                result = match_bond(
                    BondMatchRequest(
                        trade_message_id="trade-1",
                        isin_result=extract_isin_from_message(text, "trade-1"),
                    ),
                    LookupMustNotRun(),
                )
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.isin_status, phase3_status)
        self.assertEqual(lookups, [])

    def test_secondary_match_requires_exact_issuer_and_supported_corroboration(self):
        index = self.reference_index()
        text = "Synthetic issuer matures 14-Jan-2030 at 8.50%."
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                raw_issuer_text="Synthetic issuer",
                issuer_evidence=self.evidence(
                    text, "Synthetic issuer", "issuer"
                ),
                maturity_date="14-Jan-2030",
                maturity_evidence=self.evidence(
                    text, "14-Jan-2030", "maturity_date"
                ),
                coupon_rate="8.50%",
                coupon_evidence=self.evidence(text, "8.50%", "coupon_rate"),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.SECONDARY_ISSUER_MATCH)
        self.assertEqual(
            result.identity_level, "uniquely resolved by secondary details"
        )
        self.assertEqual(result.matching_method, "secondary_issuer")
        self.assertEqual(result.matched_reference.normalized_isin, "US0378331005")
        self.assertEqual(result.raw_issuer_text, "Synthetic issuer")
        self.assertEqual(result.normalized_issuer_key, "synthetic issuer")
        self.assertEqual(len(result.source_evidence), 3)

    def test_secondary_matching_is_case_and_whitespace_normalized(self):
        index = self.reference_index()
        text = "SYNTHETIC ISSUER for Synthetic Bond 2030"
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                raw_issuer_text="  synthetic   issuer ",
                issuer_evidence=self.evidence(
                    text, "SYNTHETIC ISSUER", "issuer"
                ),
                security_description="synthetic bond 2030",
                description_evidence=self.evidence(
                    text, "Synthetic Bond 2030", "security_description"
                ),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.SECONDARY_ISSUER_MATCH)
        self.assertEqual(result.matched_reference.normalized_isin, "US0378331005")
        self.assertEqual(result.raw_issuer_text, "  synthetic   issuer ")
        self.assertEqual(result.normalized_issuer_key, "synthetic issuer")

    def test_exact_issuer_without_corroboration_does_not_resolve(self):
        index = self.reference_index()
        text = "Synthetic Issuer"
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                raw_issuer_text=text,
                issuer_evidence=self.evidence(text, text, "issuer"),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.NO_MATCH)
        self.assertIsNone(result.matched_reference)
        self.assertIn("no corroborating", result.explanation)

    def test_missing_issuer_and_identifier_produces_no_match(self):
        result = match_bond(
            BondMatchRequest(trade_message_id="trade-1"),
            self.reference_index(),
        )

        self.assertEqual(result.status, BondMatchStatus.NO_MATCH)
        self.assertEqual(result.matching_method, "none")
        self.assertIsNone(result.matched_reference)

    def test_multiple_secondary_securities_are_ambiguous(self):
        index = self.reference_index(rows=[
            (
                "US0378331005",
                "Synthetic Issuer",
                "Synthetic Bond A",
                "14-Jan-2030",
                "8.5000",
                "INR",
            ),
            (
                "US5949181045",
                "Synthetic Issuer",
                "Synthetic Bond B",
                "14-Jan-2030",
                "8.5000",
                "INR",
            ),
        ])
        text = "Synthetic Issuer maturity 14-Jan-2030 coupon 8.50"
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                raw_issuer_text="Synthetic Issuer",
                issuer_evidence=self.evidence(text, "Synthetic Issuer", "issuer"),
                maturity_date="14-Jan-2030",
                maturity_evidence=self.evidence(text, "14-Jan-2030", "maturity_date"),
                coupon_rate="8.50",
                coupon_evidence=self.evidence(text, "8.50", "coupon_rate"),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.AMBIGUOUS_MATCH)
        self.assertIsNone(result.matched_reference)
        self.assertEqual(
            {record.normalized_isin for record in result.candidate_references},
            {"US0378331005", "US5949181045"},
        )

    def test_contradicted_secondary_details_are_not_used_as_a_match(self):
        index = self.reference_index()
        text = "Synthetic Issuer maturity 14-Jan-2031."
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                raw_issuer_text="Synthetic Issuer",
                issuer_evidence=self.evidence(text, "Synthetic Issuer", "issuer"),
                maturity_date="14-Jan-2031",
                maturity_evidence=self.evidence(
                    text, "14-Jan-2031", "maturity_date"
                ),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.NO_MATCH)
        self.assertIsNone(result.matched_reference)

    def test_conflicting_primary_and_secondary_evidence_selects_neither(self):
        index = self.reference_index(rows=[
            (
                "US0378331005",
                "Synthetic Primary",
                "Synthetic Primary Bond",
                "14-Jan-2030",
                "8.5000",
                "INR",
            ),
            (
                "US5949181045",
                "Synthetic Secondary",
                "Synthetic Secondary Bond",
                "15-Feb-2031",
                "9.2500",
                "INR",
            ),
        ])
        text = (
            "ISIN US0378331005; Synthetic Secondary matures "
            "15-Feb-2031 with coupon 9.25."
        )
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(text, "trade-1"),
                raw_issuer_text="Synthetic Secondary",
                issuer_evidence=self.evidence(
                    text, "Synthetic Secondary", "issuer"
                ),
                maturity_date="15-Feb-2031",
                maturity_evidence=self.evidence(
                    text, "15-Feb-2031", "maturity_date"
                ),
                coupon_rate="9.25",
                coupon_evidence=self.evidence(text, "9.25", "coupon_rate"),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.CONFLICTING_EVIDENCE)
        self.assertIsNone(result.matched_reference)
        self.assertEqual(
            {record.normalized_isin for record in result.candidate_references},
            {"US0378331005", "US5949181045"},
        )

    def test_unsupported_llm_identifier_is_not_matched(self):
        index = self.reference_index()
        unsupported = validate_llm_isin(
            "US0378331005",
            "No identifier is present in this trade.",
            original_message_id="trade-1",
        )
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=unsupported,
            ),
            index,
        )

        self.assertEqual(
            result.status,
            BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
        )
        self.assertEqual(result.isin_status, IsinStatus.UNSUPPORTED_BY_SOURCE)
        self.assertIsNone(result.matched_reference)

    def test_unrelated_issuer_message_evidence_is_rejected(self):
        index = self.reference_index()
        unrelated_text = "Synthetic Issuer"
        unrelated_evidence = BondFieldEvidence(
            field_name="issuer",
            message_id="unrelated-message",
            source_message_text=unrelated_text,
            start=0,
            end=len(unrelated_text),
            linked_to_message_id="another-trade",
            link_evidence="reply to another trade",
        )
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                raw_issuer_text=unrelated_text,
                issuer_evidence=unrelated_evidence,
                maturity_date="14-Jan-2030",
                maturity_evidence=BondFieldEvidence(
                    field_name="maturity_date",
                    message_id="trade-1",
                    source_message_text="14-Jan-2030",
                    start=0,
                    end=len("14-Jan-2030"),
                ),
            ),
            index,
        )

        self.assertEqual(
            result.status,
            BondMatchStatus.UNSUPPORTED_SECONDARY_EVIDENCE,
        )
        self.assertIsNone(result.matched_reference)

    def test_supported_primary_match_is_not_blocked_by_unrelated_secondary_text(self):
        index = self.reference_index()
        message = "US0378331005. Synthetic Issuer."
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(message, "trade-1"),
                raw_issuer_text="Synthetic Issuer",
                issuer_evidence=BondFieldEvidence(
                    field_name="issuer",
                    message_id="other-message",
                    source_message_text=message,
                    start=message.index("Synthetic Issuer"),
                    end=len(message) - 1,
                    linked_to_message_id="unrelated-trade",
                    link_evidence="unrelated reply",
                ),
            ),
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.PRIMARY_ISIN_MATCH)
        self.assertEqual(result.matched_reference.normalized_isin, "US0378331005")
        self.assertIn("Secondary details were ignored", result.explanation)

    def test_unrelated_llm_identifier_message_is_rejected_by_phase_3(self):
        with self.assertRaisesRegex(ValueError, "explicit link evidence"):
            validate_llm_isin(
                "US0378331005",
                "Trade details without an ISIN.",
                original_message_id="trade-1",
                linked_messages=(
                    LinkedMessageEvidence(
                        message_id="unrelated",
                        message_text="US0378331005",
                        linked_to_message_id="different-trade",
                        link_evidence="reply link",
                    ),
                ),
            )

    def test_isin_evidence_from_an_unrelated_message_is_not_matched(self):
        result = match_bond(
            BondMatchRequest(
                trade_message_id="trade-1",
                isin_result=extract_isin_from_message(
                    "US0378331005", "unrelated-message"
                ),
            ),
            self.reference_index(),
        )

        self.assertEqual(
            result.status,
            BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
        )
        self.assertIsNone(result.matched_reference)

    def test_unavailable_reference_index_is_explicit(self):
        result = match_bond(
            BondMatchRequest(trade_message_id="trade-1"),
            None,
        )

        self.assertEqual(result.status, BondMatchStatus.REFERENCE_UNAVAILABLE)
        self.assertIn("No bond reference index", result.explanation)

    def test_empty_csv_fails_instead_of_becoming_an_empty_index(self):
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)

        with self.assertRaisesRegex(BondReferenceError, "no usable records"):
            BondReferenceIndex.load(self.csv_path)


if __name__ == "__main__":
    unittest.main()
