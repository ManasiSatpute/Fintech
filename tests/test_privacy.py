import unittest
from unittest.mock import patch

import config
import pipeline
import privacy


class PrivacyMaskingTests(unittest.TestCase):
    def test_masks_contact_details_but_preserves_trade_values(self):
        text = (
            "Call +91 98765 43210 or @919876543210; "
            "email trader@example.com, https://example.com/rfq. "
            "HDFC Bank 1234 account no. 5678. "
            "ISIN US0378331005, yield 8.25%, qty 5 Cr, "
            "date 2026-07-13, RFQ R123456789012."
        )

        masked, placeholders = privacy.mask_pii(text)

        self.assertNotIn("+91 98765 43210", masked)
        self.assertNotIn("@919876543210", masked)
        self.assertNotIn("trader@example.com", masked)
        self.assertNotIn("https://example.com/rfq", masked)
        bare_url, bare_url_map = privacy.mask_pii("Visit example.com/trades")
        self.assertNotIn("example.com/trades", bare_url)
        self.assertEqual(
            privacy.restore_pii(bare_url, bare_url_map),
            "Visit example.com/trades",
        )
        self.assertNotIn("HDFC Bank 1234", masked)
        self.assertNotIn("account no. 5678", masked)
        self.assertIn("US0378331005", masked)
        self.assertIn("8.25%", masked)
        self.assertIn("5 Cr", masked)
        self.assertIn("2026-07-13", masked)
        self.assertIn("R123456789012", masked)
        self.assertEqual(privacy.restore_pii(masked, placeholders), text)

    def test_masks_explicit_sender_numbers_and_restores_nested_results(self):
        masked, placeholders = privacy.mask_pii(
            "Sender 919876543210 offered 5 Cr",
            sender_numbers=["919876543210:2"],
        )

        self.assertNotIn("919876543210", masked)
        result = {"deals": [{"buyer_text": "[PII_001]", "quantity_text": "5 Cr"}]}
        self.assertEqual(
            privacy.restore_pii(result, placeholders),
            {
                "deals": [{
                    "buyer_text": "919876543210",
                    "quantity_text": "5 Cr",
                }]
            },
        )

    def test_person_names_are_masked_only_when_enabled(self):
        text = "Alice offers 5 Cr to Bob"

        unmasked, empty_map = privacy.mask_pii(
            text,
            person_names=["Alice", "Bob"],
        )
        masked, placeholders = privacy.mask_pii(
            text,
            person_names=["Alice", "Bob"],
            mask_person_names=True,
        )

        self.assertIn("Alice", unmasked)
        self.assertIn("Bob", unmasked)
        self.assertEqual(empty_map, {})
        self.assertNotIn("Alice", masked)
        self.assertNotIn("Bob", masked)
        self.assertIn("5 Cr", masked)
        self.assertEqual(privacy.restore_pii(masked, placeholders), text)

    def test_context_snippet_stores_local_mapping_when_masking_is_enabled(self):
        from datetime import datetime

        messages = [{
            "message_id": "synthetic-message",
            "chat_name": "Synthetic desk",
            "chat_type": "PRIVATE",
            "sender": "Alice",
            "sender_number": "919876543210",
            "receiver_number": "",
            "is_own": False,
            "is_media": False,
            "message_text": (
                "US0378331005 5 Cr at 8.25% "
                "contact +91 98765 43210"
            ),
            "replying_to": "",
            "dt": datetime(2026, 7, 13, 10, 0),
        }]
        placeholder_map = {}

        with (
            patch.object(config, "PII_MASKING_ENABLED", True),
            patch.object(config, "MASK_PERSON_NAMES", False),
        ):
            snippet, _ = pipeline.format_context_snippet(
                messages,
                "Synthetic desk",
                placeholder_map,
            )

        self.assertNotIn("98765 43210", snippet)
        self.assertIn("US0378331005", snippet)
        self.assertIn("5 Cr", snippet)
        self.assertIn("8.25%", snippet)
        self.assertTrue(placeholder_map)
        self.assertIn("Alice", snippet)
        self.assertIn(
            "contact +91 98765 43210",
            privacy.restore_pii(snippet, placeholder_map),
        )
        names_masked_map = {}
        with (
            patch.object(config, "PII_MASKING_ENABLED", True),
            patch.object(config, "MASK_PERSON_NAMES", True),
        ):
            names_masked_snippet, _ = pipeline.format_context_snippet(
                messages,
                "Synthetic desk",
                names_masked_map,
            )
        self.assertNotIn("Alice", names_masked_snippet)
        self.assertIn(
            "Alice: US0378331005",
            privacy.restore_pii(names_masked_snippet, names_masked_map),
        )

    def test_context_snippet_does_not_mask_when_disabled(self):
        from datetime import datetime

        messages = [{
            "message_id": "synthetic-message",
            "chat_name": "Synthetic desk",
            "chat_type": "PRIVATE",
            "sender": "Alice",
            "sender_number": "919876543210",
            "receiver_number": "",
            "is_own": False,
            "is_media": False,
            "message_text": "Call 9876543210",
            "replying_to": "",
            "dt": datetime(2026, 7, 13, 10, 0),
        }]
        placeholder_map = {}

        with patch.object(config, "PII_MASKING_ENABLED", False):
            snippet, _ = pipeline.format_context_snippet(
                messages,
                "Synthetic desk",
                placeholder_map,
            )

        self.assertIn("Call 9876543210", snippet)
        self.assertEqual(placeholder_map, {})


if __name__ == "__main__":
    unittest.main()
