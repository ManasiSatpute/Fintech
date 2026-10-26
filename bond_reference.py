"""Immutable, in-memory access to the external bond reference CSV."""

import csv
import hashlib
import io
import logging
import threading
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple, Union

import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "ISIN",
    "NAME_OF_ISSUER",
    "SECURITY_DESCRIPTION",
    "MATURITY_DATE",
    "INTEREST_RATE",
    "CURRENCY",
)

_DATE_FORMATS = (
    "%d-%b-%Y",
    "%d-%b-%y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
)


class BondReferenceError(ValueError):
    """Raised when a reference CSV cannot be loaded safely."""


@dataclass(frozen=True)
class BondReferenceRecord:
    """One accepted source row, with original values kept separate from keys."""

    source_values: Mapping[str, str]
    normalized_isin: str
    normalized_issuer_key: str
    maturity_date: Optional[date]


@dataclass(frozen=True)
class BondReferenceMetadata:
    filename: str
    sha256: str
    loaded_at: str
    encoding: str
    total_source_rows: int
    successfully_loaded_records: int
    rejected_records: int
    duplicate_identifier_count: int
    identical_duplicate_identifier_count: int
    conflicting_identifier_count: int
    validation_summary: Mapping[str, int]


def normalize_isin_key(value: Optional[str]) -> str:
    """Normalize case and all whitespace without applying ISIN validation."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(normalized.split()).upper()


def normalize_issuer_key(value: Optional[str]) -> str:
    """Normalize Unicode, case and whitespace for issuer candidate retrieval."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def _parse_maturity_date(value: str) -> Optional[date]:
    stripped = value.strip()
    if not stripped:
        return None
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(stripped, date_format).date()
        except ValueError:
            continue
    raise ValueError("invalid_maturity_date")


def parse_reference_maturity_date(value: str) -> Optional[date]:
    """Parse a non-empty maturity value using the reference CSV's date formats."""
    return _parse_maturity_date(value)


def _decode_csv(raw_bytes: bytes) -> Tuple[str, str]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise BondReferenceError(
        "Reference CSV encoding is unsupported; expected UTF-8 or Windows-1252."
    )


class BondReferenceIndex:
    """A read-only index of reference records loaded from one CSV snapshot."""

    def __init__(
        self,
        records_by_isin: Mapping[str, Tuple[BondReferenceRecord, ...]],
        records_by_issuer: Mapping[str, Tuple[BondReferenceRecord, ...]],
        metadata: BondReferenceMetadata,
    ) -> None:
        self._records_by_isin = MappingProxyType(dict(records_by_isin))
        self._records_by_issuer = MappingProxyType(dict(records_by_issuer))
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        csv_path: Optional[Union[str, Path]] = None,
    ) -> "BondReferenceIndex":
        path = Path(csv_path if csv_path is not None else config.BOND_REFERENCE_CSV)
        if not path.is_file():
            logger.error(
                "Bond reference CSV unavailable: category=missing_file path=%s",
                path,
            )
            raise BondReferenceError(
                "Bond reference CSV was not found. Check BOND_REFERENCE_CSV."
            )

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            logger.error(
                "Bond reference CSV unavailable: category=read_error path=%s "
                "error_type=%s",
                path,
                type(exc).__name__,
            )
            raise BondReferenceError(
                "Bond reference CSV could not be read. Check the configured "
                "file and its permissions."
            ) from exc
        digest = hashlib.sha256(raw_bytes).hexdigest()
        try:
            text, encoding = _decode_csv(raw_bytes)
        except BondReferenceError:
            logger.error(
                "Bond reference CSV unavailable: category=unsupported_encoding "
                "path=%s",
                path,
            )
            raise
        del raw_bytes

        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        try:
            headers = next(reader, None)
        except csv.Error as exc:
            logger.error(
                "Bond reference CSV unavailable: category=invalid_header "
                "path=%s error_type=%s",
                path,
                type(exc).__name__,
            )
            raise BondReferenceError(
                "Bond reference CSV header could not be parsed."
            ) from exc
        if headers is None:
            logger.error(
                "Bond reference CSV unavailable: category=empty_file path=%s",
                path,
            )
            raise BondReferenceError("Bond reference CSV is empty.")

        normalized_headers = [header.strip() for header in headers]
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in normalized_headers
        ]
        duplicate_required_headers = [
            column for column in REQUIRED_COLUMNS
            if normalized_headers.count(column) > 1
        ]
        if missing_columns or duplicate_required_headers:
            details = []
            if missing_columns:
                details.append(f"missing required columns: {', '.join(missing_columns)}")
            if duplicate_required_headers:
                details.append(
                    "duplicate required columns: "
                    + ", ".join(duplicate_required_headers)
                )
            logger.error(
                "Bond reference CSV unavailable: category=invalid_columns "
                "path=%s details=%s",
                path,
                "; ".join(details),
            )
            raise BondReferenceError(
                f"Bond reference CSV has {'; '.join(details)}."
            )

        column_indexes = {
            column: normalized_headers.index(column) for column in REQUIRED_COLUMNS
        }
        rows_by_isin: Dict[str, list[BondReferenceRecord]] = {}
        records_by_issuer: Dict[str, list[BondReferenceRecord]] = {}
        validation_counts = {
            "malformed_rows": 0,
            "missing_isin": 0,
            "invalid_maturity_date": 0,
            "empty_issuer": 0,
        }
        total_source_rows = 0

        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                total_source_rows += 1
                validation_counts["malformed_rows"] += 1
                continue

            total_source_rows += 1
            if len(row) != len(headers):
                validation_counts["malformed_rows"] += 1
                continue

            source_values = {
                column: row[column_indexes[column]] for column in REQUIRED_COLUMNS
            }
            normalized_isin = normalize_isin_key(source_values["ISIN"])
            if not normalized_isin:
                validation_counts["missing_isin"] += 1
                continue

            try:
                maturity_date = _parse_maturity_date(
                    source_values["MATURITY_DATE"]
                )
            except ValueError:
                validation_counts["invalid_maturity_date"] += 1
                continue

            normalized_issuer = normalize_issuer_key(
                source_values["NAME_OF_ISSUER"]
            )
            if not normalized_issuer:
                validation_counts["empty_issuer"] += 1
            record = BondReferenceRecord(
                source_values=MappingProxyType(source_values),
                normalized_isin=normalized_isin,
                normalized_issuer_key=normalized_issuer,
                maturity_date=maturity_date,
            )
            rows_by_isin.setdefault(normalized_isin, []).append(record)
            if normalized_issuer:
                records_by_issuer.setdefault(normalized_issuer, []).append(record)

        successfully_loaded = sum(len(records) for records in rows_by_isin.values())
        rejected_records = total_source_rows - successfully_loaded
        if not successfully_loaded:
            logger.error(
                "Bond reference CSV unavailable: category=no_usable_records "
                "path=%s source_rows=%s rejected_records=%s",
                path,
                total_source_rows,
                rejected_records,
            )
            raise BondReferenceError(
                f"Bond reference CSV contains no usable records "
                f"({total_source_rows} source rows, {rejected_records} rejected)."
            )

        duplicate_identifier_count = 0
        identical_duplicate_identifier_count = 0
        conflicting_identifier_count = 0
        for duplicate_records in rows_by_isin.values():
            if len(duplicate_records) < 2:
                continue
            duplicate_identifier_count += 1
            signatures = {
                tuple(
                    record.source_values[column]
                    for column in REQUIRED_COLUMNS
                    if column != "ISIN"
                )
                for record in duplicate_records
            }
            if len(signatures) == 1:
                identical_duplicate_identifier_count += 1
            else:
                conflicting_identifier_count += 1

        metadata = BondReferenceMetadata(
            filename=path.name,
            sha256=digest,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            encoding=encoding,
            total_source_rows=total_source_rows,
            successfully_loaded_records=successfully_loaded,
            rejected_records=rejected_records,
            duplicate_identifier_count=duplicate_identifier_count,
            identical_duplicate_identifier_count=identical_duplicate_identifier_count,
            conflicting_identifier_count=conflicting_identifier_count,
            validation_summary=MappingProxyType(validation_counts),
        )
        return cls(
            records_by_isin={
                isin: tuple(records) for isin, records in rows_by_isin.items()
            },
            records_by_issuer={
                issuer: tuple(records) for issuer, records in records_by_issuer.items()
            },
            metadata=metadata,
        )

    def lookup_isin_candidates(
        self,
        isin: Optional[str],
    ) -> Tuple[BondReferenceRecord, ...]:
        key = normalize_isin_key(isin)
        return self._records_by_isin.get(key, ()) if key else ()

    def lookup_isin(self, isin: Optional[str]) -> Optional[BondReferenceRecord]:
        """Return an unambiguous record; conflicting duplicate identifiers return None."""
        candidates = self.lookup_isin_candidates(isin)
        if not candidates:
            return None
        signature = tuple(
            candidates[0].source_values[column]
            for column in REQUIRED_COLUMNS
            if column != "ISIN"
        )
        if any(
            tuple(
                record.source_values[column]
                for column in REQUIRED_COLUMNS
                if column != "ISIN"
            )
            != signature
            for record in candidates[1:]
        ):
            return None
        return candidates[0]

    def candidates_by_issuer(
        self,
        issuer: Optional[str],
    ) -> Tuple[BondReferenceRecord, ...]:
        key = normalize_issuer_key(issuer)
        return self._records_by_issuer.get(key, ()) if key else ()


_index_lock = threading.RLock()
_process_index: Optional[BondReferenceIndex] = None


def get_bond_reference_index() -> BondReferenceIndex:
    """Lazily load once per process and reuse the immutable index."""
    global _process_index
    with _index_lock:
        if _process_index is None:
            _process_index = BondReferenceIndex.load()
        return _process_index


def reload_bond_reference_index(
    csv_path: Optional[Union[str, Path]] = None,
) -> BondReferenceIndex:
    """Build a fresh snapshot and atomically make it the process-wide index."""
    global _process_index
    with _index_lock:
        replacement = BondReferenceIndex.load(csv_path)
        _process_index = replacement
        return replacement
