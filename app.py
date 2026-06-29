"""Provenance Guard — Flask application.

Milestone 3 scope: POST /submit (Signal 1 only, placeholder scoring/label) and
GET /log. Signal 2, real confidence scoring, transparency labels, appeals, and
rate limiting are layered in across M4-M5.
"""

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request

import db
import scoring
from signals.llm import score_llm
from signals.stylometry import score_stylometry

app = Flask(__name__)
db.init_db()


def _now_iso():
    """UTC timestamp in ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text or not creator_id:
        return jsonify({"error": "text and creator_id are required"}), 400

    content_id = str(uuid.uuid4())
    timestamp = _now_iso()

    # Run both signals, then combine into a calibrated confidence.
    llm = score_llm(text)
    stylometry = score_stylometry(text)
    result = scoring.score(llm["ai_probability"], stylometry["ai_probability"])

    ai_probability = result["ai_probability"]
    attribution = result["attribution"]
    label = "(placeholder label — full transparency label arrives in M5)"

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
            "status": "classified",
        },
    )

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": ai_probability,
        "confidence_level": result["confidence_level"],
        "signals": {"llm": llm, "stylometry": stylometry},
        "label": label,
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": db.get_log()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
