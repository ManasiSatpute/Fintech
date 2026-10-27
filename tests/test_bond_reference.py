import csv
import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import bond_reference
import db


FIELDS = (
    "ISIN",
    "NAME_OF_ISSUER",
    "SECURITY_DESCRIPTION",
    "MATURITY_DATE",
    "INTEREST_RATE",
    "CURRENCY",
)


class BondReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.csv_path = self.directory / "synthetic-reference.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, rows, fields=FIELDS, encoding="utf-8"):
        with self.csv_path.open("w", newline="", encoding=encoding) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return self.csv_path

    def write_raw_rows(self, rows, fields=FIELDS):
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return self.csv_path

    @staticmethod
    def record(isin=" XX 123 ", issuer=" Synthetic Issuer ", maturity="14-Jan-2030"):
        return {
            "ISIN": isin,
            "NAME_OF_ISSUER": issuer,
            "SECURITY_DESCRIPTION": " Synthetic security ",
            "MATURITY_DATE": maturity,
            "INTEREST_RATE": " 8.5000 ",
            "CURRENCY": " INR ",
        }

    def test_loads_source_values_and_exact_isin_lookup(self):
        self.write_csv([self.record()])

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        result = index.lookup_isin("XX123")
        self.assertIsNotNone(result)
        self.assertEqual(result.source_values["ISIN"], " XX 123 ")
        self.assertEqual(result.source_values["INTEREST_RATE"], " 8.5000 ")
        self.assertEqual(result.normalized_isin, "XX123")
        self.assertEqual(result.maturity_date, date(2030, 1, 14))

    def test_case_and_whitespace_are_normalized_for_lookup(self):
        self.write_csv([self.record(isin=" ab 12\t34 ", issuer="  Synthetic   Issuer ")])

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertIsNotNone(index.lookup_isin(" AB 1234 "))
        candidates = index.candidates_by_issuer("synthetic issuer")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_values["NAME_OF_ISSUER"], "  Synthetic   Issuer ")

    def test_missing_isin_lookup_returns_none(self):
        self.write_csv([self.record()])

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertIsNone(index.lookup_isin(None))
        self.assertIsNone(index.lookup_isin("not-present"))
        self.assertEqual(index.lookup_isin_candidates(""), ())

    def test_identical_duplicate_isins_are_counted_and_remain_lookable(self):
        duplicate = self.record(isin="DUP 1")
        equivalent_duplicate = duplicate.copy()
        equivalent_duplicate["ISIN"] = " dup1 "
        self.write_csv([duplicate, equivalent_duplicate])

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertEqual(index.metadata.duplicate_identifier_count, 1)
        self.assertEqual(index.metadata.identical_duplicate_identifier_count, 1)
        self.assertEqual(index.metadata.conflicting_identifier_count, 0)
        self.assertEqual(len(index.lookup_isin_candidates("DUP1")), 2)
        self.assertIsNotNone(index.lookup_isin("DUP1"))

    def test_conflicting_duplicate_isins_are_not_arbitrarily_selected(self):
        self.write_csv([
            self.record(isin="DUP 2", issuer="Synthetic Issuer A"),
            self.record(isin="DUP2", issuer="Synthetic Issuer B"),
        ])

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertEqual(index.metadata.duplicate_identifier_count, 1)
        self.assertEqual(index.metadata.identical_duplicate_identifier_count, 0)
        self.assertEqual(index.metadata.conflicting_identifier_count, 1)
        self.assertIsNone(index.lookup_isin("DUP2"))
        self.assertEqual(len(index.lookup_isin_candidates("DUP2")), 2)

    def test_missing_columns_are_reported(self):
        incomplete_fields = FIELDS[:-1]
        row = {field: "value" for field in incomplete_fields}
        self.write_csv([row], fields=incomplete_fields)

        with self.assertRaisesRegex(
            bond_reference.BondReferenceError, "missing required columns: CURRENCY"
        ):
            bond_reference.BondReferenceIndex.load(self.csv_path)

    def test_malformed_rows_empty_isins_and_invalid_dates_are_counted(self):
        good = self.record(isin="GOOD")
        missing_isin = self.record(isin="  ")
        invalid_date = self.record(isin="BAD-DATE", maturity="not a date")
        self.write_raw_rows([[good[field] for field in FIELDS]])
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                'BAD-QUOTE,"Synthetic"tail,Synthetic security,,8,INR\n'
            )
            writer = csv.writer(handle)
            writer.writerows([
                [missing_isin[field] for field in FIELDS],
                [invalid_date[field] for field in FIELDS],
                [*list(good[field] for field in FIELDS), "extra"],
                ["SHORT", "Synthetic issuer", "Synthetic security", "", "8"],
            ])

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertEqual(index.metadata.total_source_rows, 6)
        self.assertEqual(index.metadata.successfully_loaded_records, 1)
        self.assertEqual(index.metadata.rejected_records, 5)
        self.assertEqual(index.metadata.validation_summary["malformed_rows"], 3)
        self.assertEqual(index.metadata.validation_summary["missing_isin"], 1)
        self.assertEqual(index.metadata.validation_summary["invalid_maturity_date"], 1)
        self.assertIsNotNone(index.lookup_isin("GOOD"))

    def test_missing_file_is_reported(self):
        missing_path = self.directory / "synthetic-missing.csv"

        with self.assertLogs(bond_reference.logger, level="ERROR") as captured:
            with self.assertRaises(bond_reference.BondReferenceError) as raised:
                bond_reference.BondReferenceIndex.load(missing_path)
        self.assertIn("Check BOND_REFERENCE_CSV", str(raised.exception))
        self.assertNotIn(str(missing_path), str(raised.exception))
        self.assertIn(str(missing_path), "\n".join(captured.output))

    def test_read_exception_details_are_logged_but_not_in_public_error(self):
        self.write_csv([self.record()])
        injected_detail = (
            r"C:\synthetic\private\reference.csv credential=SYNTHETIC_SECRET"
        )

        with self.assertLogs(bond_reference.logger, level="ERROR") as captured:
            with patch.object(
                Path,
                "read_bytes",
                side_effect=PermissionError(injected_detail),
            ):
                with self.assertRaises(
                    bond_reference.BondReferenceError
                ) as raised:
                    bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertIn("permissions", str(raised.exception))
        self.assertNotIn("SYNTHETIC_SECRET", str(raised.exception))
        self.assertNotIn(injected_detail, "\n".join(captured.output))
        self.assertIn("error_type=PermissionError", "\n".join(captured.output))

    def test_metadata_includes_hash_filename_timestamp_counts_and_validation(self):
        self.write_csv([self.record(isin="META")])
        expected_hash = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()

        index = bond_reference.BondReferenceIndex.load(self.csv_path)
        metadata = index.metadata

        self.assertEqual(metadata.filename, self.csv_path.name)
        self.assertEqual(metadata.sha256, expected_hash)
        self.assertEqual(metadata.encoding, "utf-8-sig")
        self.assertIsNotNone(datetime.fromisoformat(metadata.loaded_at).tzinfo)
        self.assertEqual(metadata.total_source_rows, 1)
        self.assertEqual(metadata.successfully_loaded_records, 1)
        self.assertEqual(metadata.rejected_records, 0)
        self.assertEqual(dict(metadata.validation_summary), {
            "malformed_rows": 0,
            "missing_isin": 0,
            "invalid_maturity_date": 0,
            "empty_issuer": 0,
        })

    def test_windows_1252_encoding_is_supported(self):
        row = self.record(issuer="Synthetic Société")
        self.write_csv([row], encoding="cp1252")

        index = bond_reference.BondReferenceIndex.load(self.csv_path)

        self.assertEqual(index.metadata.encoding, "cp1252")
        self.assertEqual(
            index.lookup_isin("XX123").source_values["NAME_OF_ISSUER"],
            "Synthetic Société",
        )

    def test_reload_replaces_snapshot_only_after_successful_build(self):
        self.write_csv([self.record(isin="OLD")])
        original = bond_reference.reload_bond_reference_index(self.csv_path)
        self.assertIs(bond_reference.get_bond_reference_index(), original)

        self.write_csv([self.record(isin="NEW")])
        self.assertIs(bond_reference.get_bond_reference_index(), original)
        self.assertIsNone(bond_reference.get_bond_reference_index().lookup_isin("NEW"))

        refreshed = bond_reference.reload_bond_reference_index(self.csv_path)

        self.assertIsNot(refreshed, original)
        self.assertIs(bond_reference.get_bond_reference_index(), refreshed)
        self.assertIsNone(refreshed.lookup_isin("OLD"))
        self.assertIsNotNone(refreshed.lookup_isin("NEW"))

    def test_concurrent_reads_share_safe_immutable_indexes(self):
        self.write_csv([self.record(isin=f"THREAD-{index}") for index in range(20)])
        bond_reference.reload_bond_reference_index(self.csv_path)

        def lookup(index_number):
            index = bond_reference.get_bond_reference_index()
            record = index.lookup_isin(f"thread-{index_number}")
            return record.source_values["ISIN"] if record else None

        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(lookup, list(range(20)) * 25))

        self.assertEqual(len(results), 500)
        self.assertTrue(all(result is not None for result in results))

    def test_reference_records_are_not_inserted_into_sqlite(self):
        self.write_csv([self.record(isin="DB-CHECK")])
        db_path = self.directory / "empty-application.db"
        db.init_db(str(db_path))

        index = bond_reference.BondReferenceIndex.load(self.csv_path)
        connection = db.get_db_connection(str(db_path))
        try:
            counts = {
                table: connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                for table in (
                    "messages",
                    "bonds",
                    "transactions",
                    "identification_audit",
                )
            }
        finally:
            connection.close()

        self.assertIsNotNone(index.lookup_isin("DB-CHECK"))
        self.assertEqual(counts, {
            "messages": 0,
            "bonds": 0,
            "transactions": 0,
            "identification_audit": 0,
        })


if __name__ == "__main__":
    unittest.main()
