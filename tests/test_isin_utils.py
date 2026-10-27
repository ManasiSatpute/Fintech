import unittest

from isin_utils import (
    IsinStatus,
    LinkedMessageEvidence,
    extract_isin_from_message,
    normalize_isin_candidate,
    validate_llm_isin,
)


class IsinExtractionTests(unittest.TestCase):
    def test_extracts_and_checksum_validates_known_isin_with_source_evidence(self):
        result = extract_isin_from_message(
            "Please check US0378331005 for the trade.", "message-1"
        )

        self.assertEqual(result.status, IsinStatus.VALID_ISIN)
        self.assertEqual(result.original_identifier, "US0378331005")
        self.assertEqual(result.normalized_identifier, "US0378331005")
        self.assertEqual(len(result.candidates), 1)
        evidence = result.source_evidence[0]
        self.assertEqual(evidence.message_id, "message-1")
        self.assertEqual(evidence.matched_text, "US0378331005")
        self.assertEqual(evidence.source_message_text, "Please check US0378331005 for the trade.")
        self.assertEqual(
            evidence.source_message_text[evidence.start:evidence.end],
            evidence.matched_text,
        )

    def test_altered_check_digit_is_a_checksum_failure(self):
        result = extract_isin_from_message("ISIN: US0378331006")

        self.assertEqual(result.status, IsinStatus.CHECKSUM_FAILURE)
        self.assertEqual(result.normalized_identifier, "US0378331006")
        self.assertTrue(result.candidates[0].format_valid)
        self.assertFalse(result.candidates[0].checksum_valid)

    def test_lowercase_and_permitted_whitespace_and_hyphen_separators(self):
        self.assertEqual(normalize_isin_candidate("us-037833 1005"), "US0378331005")
        for candidate in ("us0378331005", "US 037833 1005", "US-037833-1005"):
            with self.subTest(candidate=candidate):
                result = extract_isin_from_message(candidate)
                self.assertEqual(result.status, IsinStatus.VALID_ISIN)
                self.assertEqual(result.normalized_identifier, "US0378331005")
                self.assertEqual(result.original_identifier, candidate)

    def test_missing_identifier_returns_explicit_no_identifier(self):
        result = extract_isin_from_message("Trade details without an identifier.")

        self.assertEqual(result.status, IsinStatus.NO_IDENTIFIER)
        self.assertIsNone(result.original_identifier)
        self.assertIsNone(result.normalized_identifier)
        self.assertEqual(result.candidates, ())

    def test_malformed_labeled_candidate_has_invalid_format_status(self):
        for source_text, identifier in (
            ("ISIN: 123456", "123456"),
            ("ISIN: 123", "123"),
        ):
            with self.subTest(source_text=source_text):
                result = extract_isin_from_message(source_text)
                self.assertEqual(result.status, IsinStatus.INVALID_FORMAT)
                self.assertEqual(result.original_identifier, identifier)
                self.assertEqual(result.normalized_identifier, identifier)
                self.assertFalse(result.candidates[0].format_valid)
                self.assertIsNone(result.candidates[0].checksum_valid)

    def test_unlabeled_malformed_candidate_is_not_dropped(self):
        result = extract_isin_from_message("Bond reference AB1234567")

        self.assertEqual(result.status, IsinStatus.INVALID_FORMAT)
        self.assertEqual(result.normalized_identifier, "AB1234567")

    def test_multiple_candidates_are_not_arbitrarily_selected(self):
        result = extract_isin_from_message(
            "Compare US0378331005 against US0378331006."
        )

        self.assertEqual(result.status, IsinStatus.MULTIPLE_CANDIDATES)
        self.assertIsNone(result.original_identifier)
        self.assertIsNone(result.normalized_identifier)
        self.assertEqual(
            {candidate.normalized_identifier for candidate in result.candidates},
            {"US0378331005", "US0378331006"},
        )

    def test_repeated_same_candidate_keeps_all_occurrence_evidence(self):
        result = extract_isin_from_message(
            "US0378331005 is repeated: US0378331005", "message-2"
        )

        self.assertEqual(result.status, IsinStatus.VALID_ISIN)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.source_evidence), 2)
        self.assertEqual(
            [evidence.message_id for evidence in result.source_evidence],
            ["message-2", "message-2"],
        )

    def test_llm_candidate_requires_support_from_original_message(self):
        result = validate_llm_isin(
            "US0378331005",
            "Trade text contains no identifier.",
            original_message_id="message-3",
        )

        self.assertEqual(result.status, IsinStatus.UNSUPPORTED_BY_SOURCE)
        self.assertEqual(result.original_identifier, "US0378331005")
        self.assertEqual(result.source_evidence, ())

    def test_llm_candidate_can_use_explicitly_linked_message_evidence(self):
        result = validate_llm_isin(
            "us-037833-1005",
            "Offer details without an identifier.",
            original_message_id="trade-message",
            linked_messages=(
                LinkedMessageEvidence(
                    message_id="reply-message",
                    message_text="Reply confirms US0378331005.",
                    linked_to_message_id="trade-message",
                    link_evidence="reply relation",
                ),
            ),
        )

        self.assertEqual(result.status, IsinStatus.VALID_ISIN)
        self.assertEqual(result.normalized_identifier, "US0378331005")
        self.assertEqual(len(result.source_evidence), 1)
        self.assertEqual(result.source_evidence[0].message_id, "reply-message")
        self.assertEqual(
            result.source_evidence[0].linked_to_message_id,
            "trade-message",
        )

    def test_unlinked_conversation_message_cannot_support_llm_candidate(self):
        with self.assertRaisesRegex(ValueError, "explicit link evidence"):
            validate_llm_isin(
                "US0378331005",
                "Original trade message.",
                original_message_id="trade-message",
                linked_messages=(
                    LinkedMessageEvidence(
                        message_id="other-message",
                        message_text="US0378331005",
                        linked_to_message_id="different-trade",
                        link_evidence="unrelated",
                    ),
                ),
            )

    def test_valid_and_invalid_candidates_together_remain_ambiguous(self):
        result = validate_llm_isin(
            "US0378331005",
            "Trade includes US0378331005 and US0378331006.",
        )

        self.assertEqual(result.status, IsinStatus.MULTIPLE_CANDIDATES)
        self.assertIsNone(result.normalized_identifier)
        self.assertEqual(len(result.source_evidence), 2)


if __name__ == "__main__":
    unittest.main()
