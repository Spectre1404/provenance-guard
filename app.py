"""Provenance Guard — Flask application.

Endpoints:
  POST /submit                  classify content + return transparency label
  POST /appeal                  contest a classification (status -> under_review)
  GET  /log                     recent structured audit-log entries
  POST /verify-human            earn a "Verified Human" provenance certificate
  GET  /certificate/<creator>   fetch + validate a creator's certificate
"""

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import certificates
import db
import labels
import scoring
from signals.llm import score_llm
from signals.predictability import score_predictability
from signals.stylometry import score_stylometry

# Minimum sample length (words) required to earn a provenance certificate, so
# the detection signals have enough text to be meaningful.
MIN_VERIFY_WORDS = 60

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


def run_pipeline(text):
    """Run all three detection signals and combine into a scored result.

    Returns (llm, stylometry, predictability, result). Predictability abstains
    on short text (active=False); when it abstains the scorer renormalizes to
    the two-signal weighting.
    """
    llm = score_llm(text)
    stylometry = score_stylometry(text)
    predictability = score_predictability(text)
    pred_prob = (
        predictability["ai_probability"] if predictability["active"] else None
    )
    result = scoring.score(
        llm["ai_probability"], stylometry["ai_probability"], pred_prob
    )
    return llm, stylometry, predictability, result


def valid_certificate(creator_id):
    """Return a creator's certificate dict if present and signature-valid."""
    cert = db.get_certificate(creator_id)
    if cert and cert["status"] == "valid" and certificates.verify(
        cert["creator_id"], cert["certificate_id"], cert["issued_at"],
        cert["signature"],
    ):
        return cert
    return None


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

    llm, stylometry, predictability, result = run_pipeline(text)

    ai_probability = result["ai_probability"]
    attribution = result["attribution"]
    label = labels.make_label(ai_probability)

    # A creator holding a valid provenance certificate gets a badge prefix.
    verified = valid_certificate(creator_id) is not None
    if verified:
        label = "🏅 Verified Human Creator — " + label

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
            "verified_human": verified,
            "status": "classified",
        },
    )

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": ai_probability,
        "confidence_level": result["confidence_level"],
        "verified_human": verified,
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


@app.route("/verify-human", methods=["POST"])
def verify_human():
    data = request.get_json(silent=True) or {}
    creator_id = (data.get("creator_id") or "").strip()
    sample_text = (data.get("sample_text") or "").strip()

    if not creator_id or not sample_text:
        return jsonify(
            {"error": "creator_id and sample_text are required"}
        ), 400

    if len(sample_text.split()) < MIN_VERIFY_WORDS:
        return jsonify({
            "error": f"sample_text must be at least {MIN_VERIFY_WORDS} words to "
                     "verify; please provide a longer writing sample.",
        }), 400

    # The verification gate: the live sample must score as high-confidence human.
    _, _, _, result = run_pipeline(sample_text)
    timestamp = _now_iso()

    if result["attribution"] != "likely_human":
        # Log the failed attempt (scores only — never the raw sample text).
        db.write_log(
            content_id=f"verify:{creator_id}",
            event_type="verification",
            timestamp=timestamp,
            payload={
                "creator_id": creator_id,
                "outcome": "rejected",
                "attribution": result["attribution"],
                "confidence": result["ai_probability"],
            },
        )
        return jsonify({
            "verified": False,
            "reason": "Sample did not pass human verification "
                      f"(attribution: {result['attribution']}, "
                      f"confidence score: {result['ai_probability']}).",
        }), 200

    # Issue a tamper-evident certificate.
    certificate_id = str(uuid.uuid4())
    signature = certificates.sign(creator_id, certificate_id, timestamp)
    db.save_certificate(creator_id, certificate_id, timestamp, signature)
    db.write_log(
        content_id=f"verify:{creator_id}",
        event_type="verification",
        timestamp=timestamp,
        payload={
            "creator_id": creator_id,
            "outcome": "issued",
            "certificate_id": certificate_id,
            "confidence": result["ai_probability"],
        },
    )
    return jsonify({
        "verified": True,
        "creator_id": creator_id,
        "certificate_id": certificate_id,
        "issued_at": timestamp,
        "signature": signature,
        "message": "Verified Human credential issued. It will be displayed on "
                   "your future submissions.",
    })


@app.route("/certificate/<creator_id>", methods=["GET"])
def certificate(creator_id):
    cert = db.get_certificate(creator_id)
    if cert is None:
        return jsonify({"creator_id": creator_id, "verified": False}), 404
    is_valid = certificates.verify(
        cert["creator_id"], cert["certificate_id"], cert["issued_at"],
        cert["signature"],
    )
    return jsonify({
        "creator_id": cert["creator_id"],
        "certificate_id": cert["certificate_id"],
        "issued_at": cert["issued_at"],
        "status": cert["status"],
        "verified": is_valid and cert["status"] == "valid",
        "badge": "🏅 Verified Human Creator",
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": db.get_log()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
