import unittest
from datetime import datetime

from normalize import (
    detect_partial_isins,
    extract_isins,
    is_valid_isin,
    normalize_maturity,
    normalize_quantity,
    normalize_yield_coupon,
    resolve_settlement,
)


class QuantityNormalizationTests(unittest.TestCase):
    def test_amount_units_ranges_and_flags(self):
        crore = normalize_quantity("only 2.5 cr")
        self.assertEqual(crore["amount_inr"], 25_000_000)
        self.assertTrue(crore["only"])

        lakh_range = normalize_quantity("20-30 lakh")
        self.assertEqual(lakh_range["value"], {"min": 20, "max": 30})
        self.assertEqual(
            lakh_range["amount_inr"],
            {"min": 2_000_000, "max": 3_000_000},
        )
        self.assertTrue(lakh_range["is_indicative"])

        self.assertTrue(normalize_quantity("multiple of 100 units")["multiple_of"])
        self.assertTrue(normalize_quantity("multiple of")["multiple_of"])
        self.assertTrue(normalize_quantity("10 cr only")["only"])
        self.assertTrue(normalize_quantity("entire quantum")["entire_quantum"])

    def test_units_require_face_value_and_isin_context(self):
        self.assertEqual(normalize_quantity("100 units")["amount_inr"], None)
        self.assertEqual(
            normalize_quantity("100 units", face_value=10_000)["amount_inr"],
            1_000_000,
        )
        assumed = normalize_quantity("ISIN INE123A01016, 100")
        self.assertEqual(assumed["unit"], "Units")
        self.assertTrue(assumed["units_assumed"])
        self.assertIsNone(assumed["amount_inr"])
        self.assertIsNone(normalize_quantity("maturity Aug 2026"))


class YieldNormalizationTests(unittest.TestCase):
    def test_coupon_yield_two_way_and_basis_points(self):
        result = normalize_yield_coupon(
            "8.25% Oxyzo; bid @5.20%; 8.33-32; spread 25 bps",
            issuer_aliases=["Oxyzo"],
        )
        self.assertEqual(result["coupons"][0]["value"], 8.25)
        self.assertIn(5.2, result["yields"])
        self.assertEqual(result["two_way_quotes"][0]["values"], [8.33, 8.32])
        self.assertTrue(result["ytm_is_two_way"])
        self.assertEqual(result["spreads"][0]["percent"], 0.25)
        self.assertNotIn(0.25, result["yields"])
        self.assertEqual(
            normalize_yield_coupon("can take 6.1")["yields"],
            [6.1],
        )

    def test_out_of_range_yield_is_flagged(self):
        result = normalize_yield_coupon("best 27.5")
        self.assertTrue(result["warnings"])


class SettlementNormalizationTests(unittest.TestCase):
    def test_business_days_holidays_and_relative_t0(self):
        monday = datetime(2026, 1, 5, 12, 0)
        self.assertEqual(
            resolve_settlement(
                "T+1", monday, market_holidays=["2026-01-06"]
            )["settlement_date"],
            "2026-01-07",
        )
        friday = datetime(2026, 1, 9, 12, 0)
        tom_t0 = resolve_settlement("Tom T+0", friday)
        self.assertEqual(tom_t0["settlement_date"], "2026-01-12")
        self.assertFalse(tom_t0["needs_review"])
        self.assertEqual(tom_t0["t_plus_part"]["date"], "2026-01-12")
        self.assertEqual(
            resolve_settlement("Mon VD", monday)["settlement_date"],
            "2026-01-12",
        )

    def test_ambiguous_relative_and_t_plus_are_both_returned(self):
        result = resolve_settlement("kal T+1", datetime(2026, 1, 5))
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["relative_part"]["date"], "2026-01-06")
        self.assertEqual(result["t_plus_part"]["date"], "2026-01-06")
        self.assertIsNone(resolve_settlement("", datetime(2026, 1, 5))["settlement_date"])


class MaturityAndIsinTests(unittest.TestCase):
    def test_maturity_preserves_unknown_day(self):
        self.assertEqual(
            normalize_maturity("Aug 2026"),
            {
                "maturity_text": "Aug 2026",
                "maturity_date": None,
                "maturity_ym": "2026-08",
            },
        )
        self.assertEqual(
            normalize_maturity("27 may 29")["maturity_date"],
            "2029-05-27",
        )
        self.assertEqual(
            normalize_maturity("June 27")["maturity_ym"],
            "2027-06",
        )

    def test_full_isin_validation_and_partial_fragment_detection(self):
        result = extract_isins("US0378331005 and US0378331006; ISIN 123456")
        self.assertEqual(result["valid"], ["US0378331005"])
        self.assertEqual(result["invalid"], ["US0378331006"])
        self.assertEqual(result["partial"], ["123456"])
        self.assertTrue(is_valid_isin("US0378331005"))
        self.assertEqual(detect_partial_isins("ISIN: 123456"), ["123456"])


if __name__ == "__main__":
    unittest.main()
