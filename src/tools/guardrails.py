"""
Lightweight, dependency-free safety guardrails for the action agent.

Two independent, defensive checks are provided. Both are deliberately
*high-precision* so they never mangle a legitimate Japanese announcement during
a live demo:

  scan_for_secrets(text)  -> list[str]   # names of credential patterns detected
  seen_recently(text)     -> bool        # identical message posted very recently?
  record_post(text)       -> None        # mark a message as successfully posted

Design notes:
  - No third-party imports: these run unchanged inside the Model Serving
    container and in notebooks.
  - The de-dupe cache is process-local *by design*. In a single-replica /
    scale-to-zero serving deployment (and in any one notebook session) this
    reliably stops the common "re-ran the cell / double-clicked send" double
    post. It is NOT a distributed lock; a cross-replica guarantee would instead
    query the Unity Catalog audit log. That trade-off is intentional: the guard
    must add zero external dependencies to the hot posting path.
"""

from __future__ import annotations

import hashlib
import re
import time

# ---------------------------------------------------------------------------
# 1. Secret / credential leak scanning
# ---------------------------------------------------------------------------
# Each pattern targets a specific, real credential shape. Ordinary announcement
# text — dates, URLs, @mentions, "max_tokens=2048" — must NOT match, so we avoid
# broad heuristics (e.g. any "token=..." assignment) that could block a genuine
# post mid-demo. Precision is chosen over recall on purpose.
_SECRET_PATTERNS: dict[str, "re.Pattern[str]"] = {
    "slack_webhook_url": re.compile(r"https://hooks\.slack\.com/services/\S+"),
    "teams_webhook_url": re.compile(
        r"https://[A-Za-z0-9.-]+\.webhook\.office\.com/\S+"
    ),
    "databricks_pat": re.compile(r"\bdapi[0-9a-fA-F]{32,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
}


def scan_for_secrets(text: str) -> list[str]:
    """Return the names of any credential-shaped patterns found in ``text``.

    An empty list means the text looks safe to send. This is a defense-in-depth
    output filter: the system prompt already forbids leaking secrets, and this
    enforces that in code before anything can leave the corporate perimeter.
    """
    if not text:
        return []
    return sorted(name for name, pat in _SECRET_PATTERNS.items() if pat.search(text))


# ---------------------------------------------------------------------------
# 2. Idempotency / duplicate-post suppression
# ---------------------------------------------------------------------------
_DEDUPE_WINDOW_SECONDS = 120
_recent_posts: dict[str, float] = {}


def _digest(text: str) -> str:
    """Stable content hash of a message (whitespace-normalised at the edges)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _prune(now: float, window_seconds: int) -> None:
    """Drop cache entries older than the window so it cannot grow unbounded."""
    for key, ts in list(_recent_posts.items()):
        if now - ts > window_seconds:
            del _recent_posts[key]


def seen_recently(text: str, window_seconds: int = _DEDUPE_WINDOW_SECONDS) -> bool:
    """Return True if an identical message was recorded within the window.

    This only *checks* — it does not record. Recording is done separately by
    :func:`record_post` after a send actually succeeds, so that a failed post
    followed by a legitimate retry is not mistaken for a duplicate.
    """
    now = time.monotonic()
    _prune(now, window_seconds)
    ts = _recent_posts.get(_digest(text))
    return ts is not None and (now - ts) <= window_seconds


def record_post(text: str) -> None:
    """Record that ``text`` was just posted successfully (for de-dupe)."""
    _recent_posts[_digest(text)] = time.monotonic()


def reset_dedupe_cache() -> None:
    """Clear the de-dupe cache. Intended for tests and demo resets."""
    _recent_posts.clear()
