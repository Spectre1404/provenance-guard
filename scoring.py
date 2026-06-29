"""Confidence scoring — combine the detection signals into one calibrated score.

Combines the signals via a weighted average, maps the result to one of three
attribution buckets, and derives a user-facing confidence LEVEL based on
distance from the ambiguous middle (not the raw score).

Thresholds (see planning.md):
    0.00 - 0.40  -> likely_human   (confidence: High)
    0.40 - 0.75  -> uncertain      (confidence: Low)
    0.75 - 1.00  -> likely_ai      (confidence: High)

The asymmetry is deliberate: a false positive (calling a human's work AI) is
worse than a false negative, so the "likely_ai" bucket requires a high 0.75
while "likely_human" only needs < 0.40.
"""

# Signal weights. The LLM is the stronger semantic signal; stylometry is noisier
# (especially on short text) so it carries less weight but still moves the score.
WEIGHT_LLM = 0.65
WEIGHT_STYLOMETRY = 0.35

# Bucket boundaries.
HUMAN_MAX = 0.40
AI_MIN = 0.75


def combine(llm_prob, stylometry_prob):
    """Weighted-average combination of the two signal probabilities."""
    return WEIGHT_LLM * llm_prob + WEIGHT_STYLOMETRY * stylometry_prob


def bucket(ai_probability):
    """Map a combined probability to an attribution bucket."""
    if ai_probability >= AI_MIN:
        return "likely_ai"
    if ai_probability < HUMAN_MAX:
        return "likely_human"
    return "uncertain"


def confidence_level(attribution):
    """User-facing confidence. High when far from the uncertain middle."""
    return "Low" if attribution == "uncertain" else "High"


def score(llm_prob, stylometry_prob):
    """Full scoring result for a pair of signal probabilities.

    Returns a dict with the combined ai_probability, the attribution bucket,
    and the user-facing confidence level.
    """
    ai_probability = round(combine(llm_prob, stylometry_prob), 4)
    attribution = bucket(ai_probability)
    return {
        "ai_probability": ai_probability,
        "attribution": attribution,
        "confidence_level": confidence_level(attribution),
    }
