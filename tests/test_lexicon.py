import unittest

from scripts.baseline_report import pipeline


class LexiconTests(unittest.TestCase):
    def test_matches_normalized_phrases_and_devanagari_digits_as_hints(self):
        hints = pipeline.match_lexicon(
            "DONE HAI — amount १२ Cr. Ata Kela; accpt nh krt ka?"
        )

        self.assertIn("done hai", hints["matches"]["agreement"])
        self.assertIn("ata kela", hints["matches"]["agreement"])
        self.assertIn("accpt nh krt ka", hints["matches"]["question_cues"])
        self.assertTrue(hints["has_intent_hints"])
        self.assertEqual(pipeline._normalize_lexicon_text("１２/०७"), "12 07")

    def test_ambiguous_terms_are_marked_for_context_not_as_decisions(self):
        hints = pipeline.match_lexicon("do lo s yes no")

        self.assertEqual(hints["needs_context"], ["do", "lo", "s", "yes", "no"])
        self.assertFalse(hints["has_intent_hints"])
        self.assertEqual(set(hints["matches"]), {"needs_context"})

    def test_glossary_is_compact_and_included_in_llm_prompt(self):
        glossary = pipeline.format_lexicon_glossary()

        self.assertIn("agreement:", glossary)
        self.assertIn("relative_days:", glossary)
        self.assertIn("needs_context (ambiguous; interpret only with surrounding messages)", glossary)
        self.assertIn("laga diya", glossary)


if __name__ == "__main__":
    unittest.main()
