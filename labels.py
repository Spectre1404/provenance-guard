"""Transparency label generation.

Maps a combined ai_probability to one of three plain-language labels shown to a
reader. The exact text matches planning.md verbatim. Each label makes clear the
result is an ESTIMATE, not a verdict, and (for non-human results) points to the
appeal path.

Thresholds match scoring.py:
    >= 0.75        -> high-confidence AI
    < 0.40         -> high-confidence human
    0.40 - 0.75    -> uncertain
"""

import scoring

LABELS = {
    "likely_ai": (
        "⚠️ This content shows strong signals of AI generation "
        "(confidence: High). Our automated analysis suggests it may not be "
        "human-written. This is an estimate, not a verdict — the creator can "
        "appeal this assessment."
    ),
    "likely_human": (
        "✓ This content reads as human-written (confidence: High). Our analysis "
        "found no strong signals of AI generation."
    ),
    "uncertain": (
        "❓ We couldn't confidently determine how this content was created "
        "(confidence: Low). The signals are mixed. We're showing this "
        "transparently rather than guessing."
    ),
}


def make_label(ai_probability):
    """Return the transparency label text for a combined ai_probability."""
    return LABELS[scoring.bucket(ai_probability)]
