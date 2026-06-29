"""Provenance certificate ("Verified Human" credential) — signing helpers.

A certificate is tamper-evident: it carries an HMAC-SHA256 signature over
`creator_id|certificate_id|issued_at` keyed by a server secret. Anyone holding
the secret can recompute the signature to confirm a credential was issued by
this server and has not been altered.

The secret comes from the PROVENANCE_SECRET env var; a clearly-marked default is
used for local/dev so the feature works out of the box.
"""

import hashlib
import hmac
import os

from dotenv import load_dotenv

load_dotenv()

_SECRET = os.environ.get(
    "PROVENANCE_SECRET", "dev-only-provenance-secret-change-me"
).encode("utf-8")


def sign(creator_id, certificate_id, issued_at):
    """Return the HMAC-SHA256 hex signature for a certificate's fields."""
    message = f"{creator_id}|{certificate_id}|{issued_at}".encode("utf-8")
    return hmac.new(_SECRET, message, hashlib.sha256).hexdigest()


def verify(creator_id, certificate_id, issued_at, signature):
    """Constant-time check that a certificate's signature is authentic."""
    expected = sign(creator_id, certificate_id, issued_at)
    return hmac.compare_digest(expected, signature or "")
