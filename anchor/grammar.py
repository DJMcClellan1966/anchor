"""
Optional grammar rewrite: apply rules (pattern/replacement) or external command to improve fluency.
Runs after generation; no engine dependency.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


def rewrite(text: str, config: dict[str, Any]) -> str:
    """
    Optionally rewrite text using grammar_command (stdin/stdout) or grammar_rules_path (JSON).
    If use_grammar is false or no rules/command configured, return text unchanged.
    """
    if not (config.get("use_grammar") and (text or "").strip()):
        return text or ""

    cmd = config.get("grammar_command")
    if cmd and isinstance(cmd, str) and cmd.strip():
        try:
            result = subprocess.run(
                cmd.strip().split(),
                input=text.strip(),
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        return text.strip()

    rules_path = config.get("grammar_rules_path")
    if not rules_path:
        return text.strip()
    path = Path(rules_path)
    if not path.exists() or not path.is_file():
        return text.strip()
    try:
        with open(path, encoding="utf-8") as f:
            rules = json.load(f)
    except (json.JSONDecodeError, OSError):
        return text.strip()
    if not isinstance(rules, list):
        return text.strip()
    out = text.strip()
    for item in rules:
        if not isinstance(item, dict):
            continue
        pattern = item.get("pattern")
        replacement = item.get("replacement", "")
        if pattern is None:
            continue
        try:
            out = re.sub(pattern, replacement, out)
        except (re.error, TypeError):
            continue
    return out
