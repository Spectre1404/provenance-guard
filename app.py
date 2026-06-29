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
from signals.llm import score_llm

app = Flask(__name__)
db.init_db()


def _now_iso():
    """UTC timestamp in ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# --- Placeholder scoring/label for M3; replaced by real logic in M4/M5. ---

def _placeholder_attribution(ai_probability):
    """Temporary bucket mapping until M4 adds calibrated scoring."""
    if ai_probability >= 0.75:
        return "likely_ai"
    if ai_probability < 0.40:
        return "likely_human"
    return "uncertain"


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text or not creator_id:
        return jsonify({"error": "text and creator_id are required"}), 400

    content_id = str(uuid.uuid4())
    timestamp = _now_iso()

    # Signal 1 only for now.
    llm = score_llm(text)
    ai_probability = llm["ai_probability"]
    attribution = _placeholder_attribution(ai_probability)
    confidence = ai_probability  # placeholder; real calibration in M4
    label = "(placeholder label — full transparency label arrives in M5)"

    db.save_content(
        content_id=content_id,
        creator_id=creator_id,
        text=text,
        attribution=attribution,
        confidence=confidence,
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
            "confidence": round(confidence, 4),
            "llm_score": round(llm["ai_probability"], 4),
            "llm_rationale": llm["rationale"],
            "status": "classified",
        },
    )

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": round(confidence, 4),
        "signals": {"llm": llm},
        "label": label,
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": db.get_log()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
