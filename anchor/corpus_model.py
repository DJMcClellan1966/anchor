"""
Load and use the by-product corpus model (transition matrix) as a 1-layer LM.
No neural forward pass: lookup curr_id -> log P(next|curr).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_corpus_model(path: Path) -> dict[str, Any] | None:
    """
    Load corpus_model.json. Returns dict with vocab_size, id_to_word, transition.
    transition: curr_id (str) -> {next_id (str): log_prob}.
    Returns None if file missing or invalid.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "transition" not in data:
        return None
    return data


def next_token_log_probs(curr_token_id: int, corpus_model: dict[str, Any]) -> dict[int, float]:
    """
    Return log P(next|curr) for each next token from the transition table.
    curr_token_id and returned keys are ints.
    """
    transition = corpus_model.get("transition", {})
    row = transition.get(str(curr_token_id), {})
    return {int(k): float(v) for k, v in row.items()}


def sample_next_token(
    corpus_model: dict[str, Any],
    curr_token_id: int,
    temperature: float = 1.0,
    rng: Any = None,
) -> int | None:
    """
    Sample one next token from P(next|curr) in the corpus model.
    temperature: 1.0 = unchanged; <1 sharpens, >1 flattens.
    """
    log_probs = next_token_log_probs(curr_token_id, corpus_model)
    if not log_probs:
        return None
    if rng is None:
        import random
        rng = random
    ids = list(log_probs.keys())
    log_p = [log_probs[i] for i in ids]
    if temperature != 1.0 and temperature > 0:
        logits = [lp / temperature for lp in log_p]
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        total = sum(exp_l)
        probs = [e / total for e in exp_l]
    else:
        probs = [math.exp(lp) for lp in log_p]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
    return rng.choices(ids, weights=probs, k=1)[0]
