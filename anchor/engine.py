"""
AnchorEngine: run full flow (concept -> style -> generate -> critic).
All orchestration lives here; no branching into other repos inside the loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import critic, generator, retrieval


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

    def query(self, question: str) -> tuple[str, dict[str, Any]]:
        """
        Run anchor loop: concept bundle -> style sentences -> generate -> critic.
        Returns (response_text, critic_info).
        """
        concept_bundle = retrieval.get_concept_bundle(self._engine, question)
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
        return (response_text, critic_info)
