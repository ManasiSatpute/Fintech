import csv
import tempfile
import unittest
from pathlib import Path

from bond_identification import identify_extracted_deal
from bond_matching import BondMatchStatus
from bond_reference import BondReferenceIndex


REFERENCE_COLUMNS = (
    "ISIN",
    "NAME_OF_ISSUER",
    "SECURITY_DESCRIPTION",
    "MATURITY_DATE",
    "INTEREST_RATE",
    "CURRENCY",
)


class BondIdentificationIntegrationTests(unittest.TestCase):
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
                    "Synthetic Security A",
                    "14-Jan-2030",
                    "8.50%",
                    "USD",
                ),
            ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(REFERENCE_COLUMNS)
            writer.writerows(rows)
        return BondReferenceIndex.load(self.csv_path)

    @staticmethod
    def identify(source, deal, index):
        message = {
            "message_id": "synthetic-message",
            "message_text": source,
        }
        return identify_extracted_deal(
            deal,
            [message],
            {"#1": message["message_id"]},
            index,
        )

    def test_valid_source_isin_matches_exact_reference_and_leaves_price_null(self):
        result = self.identify(
            "Synthetic trade US0378331005",
            {
                "bond_identity_text": "US0378331005",
                "price": None,
                "evidence": {"bond_identity_text": ["#1"]},
            },
            self.reference_index(),
        )

        self.assertEqual(result.status, BondMatchStatus.PRIMARY_ISIN_MATCH)
        self.assertEqual(result.matched_reference.normalized_isin, "US0378331005")
        self.assertIsNone(result.raw_issuer_text)

    def test_invalid_checksum_is_not_used_for_reference_lookup(self):
        result = self.identify(
            "Synthetic trade US0378331006",
            {
                "partial_isin_text": "US0378331006",
                "evidence": {"partial_isin_text": ["#1"]},
            },
            self.reference_index(),
        )

        self.assertEqual(
            result.status,
            BondMatchStatus.INVALID_OR_UNSUPPORTED_IDENTIFIER,
        )
        self.assertIsNone(result.matched_reference)

    def test_missing_isin_uses_supported_exact_issuer_and_coupon(self):
        result = self.identify(
            "Synthetic Issuer coupon 8.50%",
            {
                "issuer_text": "Synthetic Issuer",
                "coupon_rate": 8.5,
                "evidence": {
                    "issuer_text": ["#1"],
                    "coupon_rate": ["#1"],
                },
                "price": None,
            },
            self.reference_index(),
        )

        self.assertEqual(result.status, BondMatchStatus.SECONDARY_ISSUER_MATCH)
        self.assertEqual(result.matched_reference.normalized_isin, "US0378331005")
        self.assertEqual(result.raw_issuer_text, "Synthetic Issuer")

    def test_multiple_source_isins_are_ambiguous(self):
        result = self.identify(
            "US0378331005 and US5949181045",
            {
                "bond_identity_text": "US0378331005 and US5949181045",
                "evidence": {"bond_identity_text": ["#1"]},
            },
            self.reference_index(),
        )

        self.assertEqual(result.status, BondMatchStatus.AMBIGUOUS_MATCH)
        self.assertIsNone(result.matched_reference)

    def test_conflicting_source_issuer_and_isin_remain_unmatched(self):
        index = self.reference_index(rows=[
            (
                "US0378331005",
                "Synthetic Issuer A",
                "Synthetic Security A",
                "14-Jan-2030",
                "8.50%",
                "USD",
            ),
            (
                "US5949181045",
                "Synthetic Issuer B",
                "Synthetic Security B",
                "15-Feb-2031",
                "9.25%",
                "USD",
            ),
        ])
        result = self.identify(
            "US0378331005 Synthetic Issuer B coupon 9.25%",
            {
                "bond_identity_text": "US0378331005",
                "issuer_text": "Synthetic Issuer B",
                "coupon_rate": 9.25,
                "evidence": {
                    "bond_identity_text": ["#1"],
                    "issuer_text": ["#1"],
                    "coupon_rate": ["#1"],
                },
            },
            index,
        )

        self.assertEqual(result.status, BondMatchStatus.CONFLICTING_EVIDENCE)
        self.assertIsNone(result.matched_reference)

    def test_missing_or_uncorroborated_identity_stays_unresolved(self):
        result = self.identify(
            "Synthetic trade discussion",
            {"evidence": {}, "price": None},
            self.reference_index(),
        )

        self.assertEqual(result.status, BondMatchStatus.NO_MATCH)
        self.assertIsNone(result.matched_reference)

    def test_missing_reference_index_is_explicit(self):
        result = self.identify(
            "Synthetic trade US0378331005",
            {
                "bond_identity_text": "US0378331005",
                "evidence": {"bond_identity_text": ["#1"]},
            },
            None,
        )

        self.assertEqual(result.status, BondMatchStatus.REFERENCE_UNAVAILABLE)
        self.assertIsNone(result.matched_reference)


if __name__ == "__main__":
    unittest.main()
