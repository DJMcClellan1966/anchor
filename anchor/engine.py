"""
AnchorEngine: run full flow (concept -> style -> generate -> critic).
All orchestration lives here; no branching into other repos inside the loop.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from . import critic, generator, retrieval


def _chunk_response(text: str, max_chunk_size: int = 120) -> list[str]:
    """Split response text into chunks on sentence boundaries; fallback to max_chunk_size."""
    if not (text or "").strip():
        return []
    text = text.strip()
    chunks: list[str] = []
    # Split on sentence boundaries: ". " or newline
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chunk_size:
            chunks.append(part)
        else:
            for i in range(0, len(part), max_chunk_size):
                chunk = part[i : i + max_chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
    return chunks


class AnchorEngine:
    """Orchestrates concept retrieval, style sentences, generation, and dictionary critic."""

    def __init__(
        self,
        engine: Any,
        config: dict[str, Any],
        generator_kind: str = "stub",
    ) -> None:
        self._engine = engine
        self._config = config
        self._generator_kind = generator_kind

    def query(
        self,
        question: str,
        return_extras: bool = False,
        stream: bool = False,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None] | Iterator[tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]]:
        """
        Run anchor loop: concept bundle -> style sentences -> generate -> critic.
        When generator_kind is graph_attention, this is the Graph LLM forward pass
        (activate -> traverse -> pattern -> refine -> critic).
        When stream=False: returns (response_text, critic_info, extras_or_None). extras is a dict with
        concept_bundle and style_sentences when return_extras is True, else None.
        When stream=True: returns an iterator that yields (chunk, None, None) for each chunk,
        then (None, critic_info, extras_or_None) as the final yield.
        """
        effective_query = self._build_effective_query(question, conversation_history)
        concept_bundle = retrieval.get_concept_bundle(self._engine, effective_query)
        use_style = self._config.get("use_style_sentences", True)
        if not use_style:
            style_sentences = []
        else:
            data_dir = self._config.get("align_data_dir") or self._config.get("ANCHOR_DATA_DIR")
            genre_id = self._config.get("default_genre_id", "retirement")
            register = self._config.get("register")
            data_path = Path(data_dir) if data_dir and str(data_dir).strip() else None
            use_graph = self._config.get("use_corpus_graph", True) and data_path and (data_path / "corpus" / "graph.json").exists()
            if use_graph:
                style_sentences = retrieval.get_style_sentences_from_graph(
                    data_path, concept_bundle, genre_id=genre_id
                )
            else:
                style_sentences = retrieval.get_style_sentences(
                    self._engine,
                    data_path,
                    concept_bundle,
                    genre_id=genre_id,
                    register=register,
                )
        response_text = generator.generate(
            question,
            concept_bundle,
            style_sentences,
            self._config,
            generator_kind=self._generator_kind,
        )
        use_critic = self._config.get("use_critic", True)
        if use_critic:
            critic_info = critic.score_and_decide(response_text, self._engine, self._config)
        else:
            critic_info = {"score": 1.0, "num_grounded": 0, "num_content": 0, "decision": "accept", "show_warning": False, "message": "Critic disabled."}
        extras = {"concept_bundle": concept_bundle, "style_sentences": style_sentences} if return_extras else None

        if stream:
            return self._query_stream(response_text, critic_info, extras)
        if return_extras:
            return (response_text, critic_info, extras)
        return (response_text, critic_info, None)

    def _query_stream(
        self, response_text: str, critic_info: dict[str, Any], extras: dict[str, Any] | None
    ) -> Iterator[tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]]:
        """Yield response chunks then (None, critic_info, extras)."""
        max_chunk = int(self._config.get("streaming_max_chunk_chars", 120))
        chunks = _chunk_response(response_text, max_chunk_size=max_chunk)
        for chunk in chunks:
            yield (chunk, None, None)
        yield (None, critic_info, extras)

    def _build_effective_query(self, question: str, conversation_history: list[tuple[str, str]] | None) -> str:
        """Build effective query from system_prompt + conversation_history + question."""
        parts: list[str] = []
        system_prompt = (self._config.get("system_prompt") or "").strip()
        if system_prompt:
            parts.append(system_prompt)
        if conversation_history:
            limit = int(self._config.get("conversation_turn_limit", 2))
            for q, a in conversation_history[-limit:]:
                if (q or "").strip():
                    parts.append("Q: " + (q or "").strip())
                if (a or "").strip():
                    parts.append("A: " + (a or "").strip())
        parts.append((question or "").strip())
        return " ".join(p for p in parts if p)
