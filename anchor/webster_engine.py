"""
Webster dictionary adapter: load a word->definition JSON and expose get_context_for_description.
Compatible with retrieval.get_concept_bundle and critic.terms_in_graph.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _tokenize(query: str) -> list[str]:
    """Return lowercase alphanumeric tokens from query."""
    return re.findall(r"[a-zA-Z][a-zA-Z0-9]*", (query or "").strip())


class WebsterEngine:
    """Dictionary engine backed by a Webster-style JSON: {"word": "definition" or "definition_list", ...}.
    Values may be a single string or a list of strings (multiple senses)."""

    def __init__(self, json_path: str | Path) -> None:
        self._path = Path(json_path).resolve()
        self._data: dict[str, str | list[str]] | None = None

    def _load(self) -> dict[str, str | list[str]]:
        if self._data is not None:
            return self._data
        if not self._path.exists():
            self._data = {}
            return self._data
        with open(self._path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            self._data = {}
            return self._data
        self._data = {}
        for k, v in raw.items():
            if isinstance(k, str) and (isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v))):
                self._data[k.lower()] = v
        return self._data

    def get_context_for_description(self, query: str) -> dict[str, Any]:
        """
        Return concept bundle shape: definition_map, key_words, definitions
        for terms found by tokenizing query and looking up each token.
        definitions values may be str or list[str] (multiple senses).
        """
        data = self._load()
        if not data or not (query or "").strip():
            return {}
        tokens = _tokenize(query)
        definition_map: dict[str, str | list[str]] = {}
        for t in tokens:
            key = t.lower()
            if key in data:
                definition_map[t] = data[key]
        if not definition_map:
            return {}
        key_words = [{"name": term} for term in definition_map]
        return {
            "definition_map": definition_map,
            "definitions": dict(definition_map),
            "key_words": key_words,
        }
