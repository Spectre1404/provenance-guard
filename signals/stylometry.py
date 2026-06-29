"""Signal 2 — Stylometric heuristics (structural).

Pure Python, no external libraries. Computes six metrics across three families
(rhythm, vocabulary, voice), normalizes each to a 0-1 "AI-likeness" sub-score,
and combines them (weighted average) into a single structural ai_probability.

These reference ranges are HEURISTIC, not trained — they encode rough
human/AI tendencies, not a learned model. Stylometry is blind to meaning and
noisy on short text, so downstream it carries the lower weight (0.35) and its
confidence is damped toward 0.5 when the text is very short.

Output: {"ai_probability": float 0-1, "metrics": {...}}
"""

import re
import statistics

# --- Lexical lists for metrics 4 and 5 -------------------------------------

# Metric 4: words/phrases LLMs overuse (the AI "register").
AI_MARKERS = [
    "furthermore", "moreover", "additionally", "however", "nevertheless",
    "consequently", "delve", "crucial", "essential", "vital", "pivotal",
    "it is important to note", "it is worth noting", "tapestry", "leverage",
    "multifaceted", "seamless", "seamlessly", "realm", "underscore",
    "underscores", "robust", "navigate", "navigating", "landscape",
    "paradigm", "stakeholders", "foster", "fostering", "intricate",
    "comprehensive", "holistic", "myriad", "testament", "ever-evolving",
    "in conclusion", "in summary",
]

# Metric 5: markers of casual human voice.
HUMAN_MARKERS = [
    "honestly", "kinda", "gonna", "wanna", "ok so", "okay so", "lol", "tbh",
    "like,", "i mean", "you know", "basically", "literally", "anyway",
    "whatever", "ugh", "yeah", "nah", "stuff", "kind of", "sort of",
]

CONTRACTIONS = re.compile(
    r"\b\w+'(t|s|re|ve|ll|d|m)\b", re.IGNORECASE
)

PUNCT_TYPES = ["—", "–", "…", ";", ":", "(", ")", "!", "?", "\"", "'", "-"]


# --- Tokenization helpers --------------------------------------------------

def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _words(text):
    return re.findall(r"[A-Za-z']+", text)


def _clamp(x):
    return max(0.0, min(1.0, x))


# --- Individual metric -> AI-likeness sub-score (0-1) ----------------------

def _burstiness_score(sentences):
    """Coefficient of variation of sentence lengths. Low variation -> AI."""
    lengths = [len(_words(s)) for s in sentences]
    if len(lengths) < 2 or statistics.mean(lengths) == 0:
        return 0.5, 0.0
    cv = statistics.pstdev(lengths) / statistics.mean(lengths)
    # cv ~0.1 (very uniform) -> AI-like ~1.0; cv ~0.7+ (bursty) -> human ~0.0
    score = _clamp((0.7 - cv) / 0.6)
    return score, round(cv, 3)


def _ttr_score(words):
    """Type-token ratio. Mid-range diversity is AI-typical."""
    if not words:
        return 0.5, 0.0
    ttr = len(set(w.lower() for w in words)) / len(words)
    # Peak AI-likeness around ttr ~0.6; extremes (very repetitive or very
    # varied) read more human. Triangular kernel centered at 0.6.
    score = _clamp(1.0 - abs(ttr - 0.6) / 0.35)
    return score, round(ttr, 3)


def _punct_variety_score(text, words):
    """Number of DISTINCT punctuation types used. Low variety -> AI."""
    distinct = sum(1 for p in PUNCT_TYPES if p in text)
    # 0-1 distinct types -> AI-like; 5+ distinct -> human-like.
    score = _clamp((4 - distinct) / 4)
    return score, distinct


def _ai_marker_score(text, words):
    """Density of AI-register words/phrases per 100 words. High -> AI."""
    if not words:
        return 0.5, 0.0
    low = text.lower()
    hits = sum(low.count(m) for m in AI_MARKERS)
    per100 = hits / len(words) * 100
    # ~2 markers per 100 words is strongly AI-like.
    score = _clamp(per100 / 2.0)
    return score, round(per100, 2)


def _human_marker_score(text, words):
    """Density of casual human markers + contractions. High -> human."""
    if not words:
        return 0.5, 0.0
    low = text.lower()
    hits = sum(low.count(m) for m in HUMAN_MARKERS)
    hits += len(CONTRACTIONS.findall(text))
    per100 = hits / len(words) * 100
    # Map human density to an AI-likeness score (inverse): ~3/100 -> human.
    score = _clamp(1.0 - per100 / 3.0)
    return score, round(per100, 2)


def _opener_diversity_score(sentences):
    """Unique sentence-opening words / sentence count. Low -> AI (formulaic)."""
    if len(sentences) < 2:
        return 0.5, 0.0
    openers = []
    for s in sentences:
        w = _words(s)
        if w:
            openers.append(w[0].lower())
    if not openers:
        return 0.5, 0.0
    diversity = len(set(openers)) / len(openers)
    # diversity ~1.0 (all unique) -> human; ~0.4 (repetitive) -> AI.
    score = _clamp((0.9 - diversity) / 0.5)
    return score, round(diversity, 3)


# Weights lean on the strongest discriminators (lexical tells + rhythm).
WEIGHTS = {
    "burstiness": 0.22,
    "ttr": 0.12,
    "punct_variety": 0.12,
    "ai_markers": 0.24,
    "human_markers": 0.20,
    "opener_diversity": 0.10,
}


def score_stylometry(text):
    """Return Signal 2's structural assessment of `text`."""
    sentences = _split_sentences(text)
    words = _words(text)

    b_score, b_raw = _burstiness_score(sentences)
    t_score, t_raw = _ttr_score(words)
    p_score, p_raw = _punct_variety_score(text, words)
    a_score, a_raw = _ai_marker_score(text, words)
    h_score, h_raw = _human_marker_score(text, words)
    o_score, o_raw = _opener_diversity_score(sentences)

    combined = (
        WEIGHTS["burstiness"] * b_score
        + WEIGHTS["ttr"] * t_score
        + WEIGHTS["punct_variety"] * p_score
        + WEIGHTS["ai_markers"] * a_score
        + WEIGHTS["human_markers"] * h_score
        + WEIGHTS["opener_diversity"] * o_score
    )

    # Short-text damping: with < 3 sentences the stats are unreliable, so pull
    # the signal toward 0.5 (uncertain) proportionally.
    n_sent = len(sentences)
    if n_sent < 3:
        damp = n_sent / 3.0  # 0, 1/3, 2/3
        combined = 0.5 + (combined - 0.5) * damp

    return {
        "ai_probability": round(_clamp(combined), 4),
        "metrics": {
            "burstiness_cv": b_raw,
            "type_token_ratio": t_raw,
            "punct_variety": p_raw,
            "ai_markers_per_100w": a_raw,
            "human_markers_per_100w": h_raw,
            "opener_diversity": o_raw,
            "sentence_count": n_sent,
            "short_text_damped": n_sent < 3,
        },
    }


if __name__ == "__main__":
    import json
    samples = {
        "clearly_ai": (
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that while the benefits "
            "of AI are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment."
        ),
        "clearly_human": (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after. my friend got "
            "the spicy version and said it was better."
        ),
    }
    for label, sample in samples.items():
        print(f"\n=== {label} ===")
        print(json.dumps(score_stylometry(sample), indent=2))
