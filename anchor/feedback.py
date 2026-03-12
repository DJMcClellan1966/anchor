"""
Feedback logging and feedback-driven visit boost for Graph LLM adaptation.
Record accept/reject per response; optionally load feedback_weights and boost sentence_visits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def record(
    session_id: str,
    query: str,
    response: str,
    accepted: bool,
    path: Path | None = None,
) -> None:
    """Append one feedback record to feedback.jsonl (or path)."""
    payload = {
        "session_id": session_id,
        "query": (query or "").strip(),
        "response": (response or "").strip(),
        "accepted": bool(accepted),
        "query_hash": hashlib.sha256((query or "").strip().encode()).hexdigest()[:16],
        "response_hash": hashlib.sha256((response or "").strip().encode()).hexdigest()[:16],
    }
    if path is None:
        path = Path("data") / "feedback.jsonl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_weights(path: Path | None = None) -> dict[str, list[int]]:
    """
    Load feedback_weights.json: query_key -> [sentence_id, ...].
    Returns {} if file missing or invalid.
    """
    if path is None:
        path = Path("data") / "feedback_weights.json"
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, list[int]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            result[str(k).strip().lower()[:200]] = [int(x) for x in v if isinstance(x, (int, float))]
    return result


def apply_boosts(
    sentence_visits: dict[int, float],
    feedback_weights: dict[str, list[int]],
    query: str,
    boost: float = 0.5,
) -> None:
    """
    In-place: boost sentence_visits[sid] by boost for each sid listed under the query key.
    query_key = query.strip().lower()[:200]. If that key is in feedback_weights, add boost to each sid.
    """
    if not feedback_weights or not query:
        return
    key = (query or "").strip().lower()[:200]
    sids = feedback_weights.get(key)
    if not sids:
        return
    for sid in sids:
        if isinstance(sid, int):
            sentence_visits[sid] = sentence_visits.get(sid, 0.0) + boost
