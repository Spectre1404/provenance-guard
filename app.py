"""Provenance Guard — Flask application.

Milestone 3 scope: POST /submit (Signal 1 only, placeholder scoring/label) and
GET /log. Signal 2, real confidence scoring, transparency labels, appeals, and
rate limiting are layered in across M4-M5.
"""

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import db
import labels
import scoring
from signals.llm import score_llm
from signals.predictability import score_predictability
from signals.stylometry import score_stylometry

app = Flask(__name__)
db.init_db()

# Rate limiting. Limits are applied per-route (see /submit). Keyed by client IP.
# In-memory storage is sufficient for local/dev use.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


def _now_iso():
    """UTC timestamp in ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text or not creator_id:
        return jsonify({"error": "text and creator_id are required"}), 400

    content_id = str(uuid.uuid4())
    timestamp = _now_iso()

    # Run all three signals, then combine into a calibrated confidence.
    # Predictability abstains on short text (active=False); when it abstains we
    # pass None so the scorer renormalizes to the two-signal weighting.
    llm = score_llm(text)
    stylometry = score_stylometry(text)
    predictability = score_predictability(text)
    pred_prob = (
        predictability["ai_probability"] if predictability["active"] else None
    )
    result = scoring.score(
        llm["ai_probability"],
        stylometry["ai_probability"],
        pred_prob,
    )

    ai_probability = result["ai_probability"]
    attribution = result["attribution"]
    label = labels.make_label(ai_probability)

    db.save_content(
        content_id=content_id,
        creator_id=creator_id,
        text=text,
        attribution=attribution,
        confidence=ai_probability,
        status="classified",
        created_at=timestamp,
    )
    db.write_log(
        content_id=content_id,
        event_type="classification",
        timestamp=timestamp,
        payload={
            "creator_id": creator_id,
            "attribution": attribution,
            "confidence": ai_probability,
            "confidence_level": result["confidence_level"],
            "llm_score": round(llm["ai_probability"], 4),
            "llm_rationale": llm["rationale"],
            "stylometry_score": stylometry["ai_probability"],
            "stylometry_metrics": stylometry["metrics"],
            "predictability_score": predictability["ai_probability"],
            "predictability_metrics": predictability["metrics"],
            "status": "classified",
        },
    )

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": ai_probability,
        "confidence_level": result["confidence_level"],
        "signals": {
            "llm": llm,
            "stylometry": stylometry,
            "predictability": predictability,
        },
        "label": label,
    })


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = (data.get("content_id") or "").strip()
    reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id or not reasoning:
        return jsonify(
            {"error": "content_id and creator_reasoning are required"}
        ), 400

    content = db.get_content(content_id)
    if content is None:
        return jsonify({"error": f"unknown content_id: {content_id}"}), 404

    timestamp = _now_iso()
    db.update_status(content_id, "under_review")

    # Log the appeal alongside the original decision so a reviewer has context.
    db.write_log(
        content_id=content_id,
        event_type="appeal",
        timestamp=timestamp,
        payload={
            "creator_id": content["creator_id"],
            "appeal_reasoning": reasoning,
            "status": "under_review",
            "original_attribution": content["attribution"],
            "original_confidence": content["confidence"],
        },
    )

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": "Your appeal has been received and the content is now "
                   "under review.",
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": db.get_log()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
