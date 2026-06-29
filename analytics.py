"""Analytics aggregation for the dashboard.

Reads the structured audit log from SQLite and computes:
  - detection patterns: count + % per attribution bucket
  - appeal rate: appeals / classifications
  - additional metrics: average confidence (overall + per bucket) and the
    verification issuance rate (certificates issued vs rejected)

All aggregation is read-only over the existing audit_log, so the dashboard never
affects classification.
"""

import db

BUCKETS = ["likely_ai", "uncertain", "likely_human"]


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def compute_analytics():
    """Return the full analytics payload as a dict."""
    entries = db.get_log(limit=100000)

    classifications = [e for e in entries if e["event_type"] == "classification"]
    appeals = [e for e in entries if e["event_type"] == "appeal"]
    verifications = [e for e in entries if e["event_type"] == "verification"]

    total = len(classifications)

    # Detection patterns + per-bucket confidence.
    by_bucket = {b: [] for b in BUCKETS}
    for e in classifications:
        bucket = e.get("attribution")
        if bucket in by_bucket:
            by_bucket[bucket].append(e.get("confidence", 0.0))

    detection_patterns = {}
    for b in BUCKETS:
        scores = by_bucket[b]
        detection_patterns[b] = {
            "count": len(scores),
            "percent": _pct(len(scores), total),
            "avg_confidence": round(sum(scores) / len(scores), 4)
            if scores else None,
        }

    all_scores = [c.get("confidence", 0.0) for c in classifications]
    avg_confidence = round(sum(all_scores) / len(all_scores), 4) \
        if all_scores else None

    # Verification issuance rate.
    issued = sum(1 for v in verifications if v.get("outcome") == "issued")
    rejected = sum(1 for v in verifications if v.get("outcome") == "rejected")
    attempts = issued + rejected

    return {
        "total_classifications": total,
        "detection_patterns": detection_patterns,
        "appeals": {
            "count": len(appeals),
            "appeal_rate_percent": _pct(len(appeals), total),
        },
        "avg_confidence_overall": avg_confidence,
        "verification": {
            "issued": issued,
            "rejected": rejected,
            "attempts": attempts,
            "issuance_rate_percent": _pct(issued, attempts),
        },
    }
