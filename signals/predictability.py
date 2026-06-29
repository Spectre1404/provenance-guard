"""Signal 3 — Predictability (information redundancy).

A pure-Python proxy for the low "perplexity" of AI text. AI-generated prose
tends to be more predictable and locally repetitive than human writing; this
signal captures redundancy that neither the semantic LLM nor the surface
stylometric metrics measure directly.

PRIVACY GUARDRAIL: this signal is computed ENTIRELY LOCALLY (zlib + n-gram
counting). Unlike the LLM signal, it transmits no text to any external service,
so it adds detection power without any additional privacy exposure.

Two sub-metrics (mappings centered so typical prose reads as ~0.5 neutral):
  1. Compression ratio (zlib): more predictable/repetitive text compresses
     better -> lower ratio -> more AI-like.
  2. Distinct n-gram ratio (bigram + trigram diversity): repetitive phrasing
     yields fewer distinct n-grams -> more AI-like.

ABSTENTION: compression/redundancy only discriminates on longer text. Below
~100 words AI and human prose look near-identical (both maximally varied), so
the signal marks itself `active: false` and the scorer drops it, renormalizing
to the two-signal weighting rather than letting a no-information reading dilute
a confident verdict.

Output: {"ai_probability": float 0-1, "active": bool, "metrics": {...}}
"""

import re
import zlib

# Minimum word count for compression/redundancy to carry real signal.
MIN_WORDS_ACTIVE = 100

# Reference points for "typical" English prose, used to center the mappings so
# ordinary varied writing maps to ~0.5 rather than to a confident human verdict.
TYPICAL_COMPRESSION = 0.66
TYPICAL_NGRAM_DISTINCT = 0.97


def _clamp(x):
    return max(0.0, min(1.0, x))


def _words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def _compression_score(text):
    """Deviation below the typical compression ratio -> more AI-like."""
    raw = text.encode("utf-8")
    if len(raw) < 40:
        return None, 0.0
    ratio = len(zlib.compress(raw, 9)) / len(raw)
    # Centered: ratio == TYPICAL -> 0.5; lower (more compressible) -> AI.
    score = _clamp(0.5 + (TYPICAL_COMPRESSION - ratio) * 3.0)
    return score, round(ratio, 3)


def _ngram_distinct_score(words, n):
    """Distinct n-gram ratio below typical (more repetition) -> more AI-like."""
    if len(words) < n + 1:
        return None, None
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    distinct_ratio = len(set(ngrams)) / len(ngrams)
    # Centered: distinct == TYPICAL -> 0.5; lower (repetitive) -> AI.
    score = _clamp(0.5 + (TYPICAL_NGRAM_DISTINCT - distinct_ratio) * 1.6)
    return score, round(distinct_ratio, 3)


def score_predictability(text):
    """Return Signal 3's predictability assessment of `text`."""
    words = _words(text)
    n_words = len(words)
    active = n_words >= MIN_WORDS_ACTIVE

    comp_score, comp_ratio = _compression_score(text)
    bi_score, bi_ratio = _ngram_distinct_score(words, 2)
    tri_score, tri_ratio = _ngram_distinct_score(words, 3)

    parts = [s for s in (comp_score, bi_score, tri_score) if s is not None]
    combined = sum(parts) / len(parts) if parts else 0.5

    return {
        "ai_probability": round(_clamp(combined), 4),
        # When inactive, the scorer ignores this signal entirely.
        "active": active,
        "metrics": {
            "compression_ratio": comp_ratio,
            "distinct_bigram_ratio": bi_ratio,
            "distinct_trigram_ratio": tri_ratio,
            "word_count": n_words,
            "abstained_short_text": not active,
        },
    }


if __name__ == "__main__":
    import json
    samples = {
        "short_ai": (
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that the benefits are "
            "numerous. Furthermore, stakeholders must collaborate."
        ),
        "long_repetitive_ai": (
            "It is important to note that artificial intelligence is "
            "transformative. It is important to note that artificial "
            "intelligence is essential. Furthermore, stakeholders must "
            "collaborate to ensure responsible deployment. Furthermore, "
            "stakeholders must collaborate across various sectors. Moreover, "
            "the ethical implications are crucial. Moreover, the ethical "
            "implications are essential to consider carefully. In conclusion, "
            "artificial intelligence represents a paradigm shift. In conclusion, "
            "the responsible deployment of artificial intelligence is vital for "
            "modern society and its many diverse stakeholders going forward."
        ),
    }
    for label, sample in samples.items():
        print(f"\n=== {label} ===")
        print(json.dumps(score_predictability(sample), indent=2))
