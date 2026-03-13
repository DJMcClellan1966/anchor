"""
Anchor math module: doc-aligned API for the unified mathematical model.

Exposes the same names as in docs/UNIFIED_MATH_MODEL.md (V, τ, S, t_s, S(w), P, J,
Embed_anchor, Propagation_layer, Output_head, Refine) and delegates to
corpus_graph and graph_attention.
"""
from __future__ import annotations

from typing import Any

from .corpus_graph import CorpusGraph
from . import graph_attention


class AnchorMath:
    """
    Wrapper holding graph, vocabulary maps, and optional encoded_index/concept_bundle.
    Exposes math-named accessors and pipeline steps that delegate to graph and graph_attention.
    """

    def __init__(
        self,
        graph: CorpusGraph,
        word_to_id: dict[str, int],
        id_to_word: dict[int, str],
        encoded_index: dict[int, dict[str, Any]] | None = None,
        concept_bundle: dict[str, Any] | None = None,
    ):
        self.graph = graph
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.encoded_index = encoded_index or {}
        self.concept_bundle = concept_bundle or {}

    # --- Vocabulary (τ, V) ---

    @property
    def V(self) -> set[int]:
        """V = set of word IDs (vocabulary)."""
        return set(self.word_to_id.values())

    def tau(self, w: str) -> int | None:
        """τ(w): word string → word ID. Returns None if not in vocab."""
        return self.word_to_id.get(w)

    def tau_inv(self, i: int) -> str | None:
        """τ⁻¹(i): word ID → word string. Returns None if unknown."""
        return self.id_to_word.get(i)

    # --- Sentence set and sentence token sequence ---

    def S(self) -> list[int]:
        """S = sentence IDs (graph.sentence_ids())."""
        return self.graph.sentence_ids()

    def t_s(self, s: int) -> list[int]:
        """t_s(s): token ID sequence for sentence s."""
        return self.graph.sentence_token_ids(s)

    def W_of_s(self, s: int) -> set[int]:
        """W(s): set of word IDs in sentence s."""
        return set(self.graph.sentence_token_ids(s))

    # --- Word → sentences S(w) ---

    def S_of_w(self, w: int) -> list[int]:
        """S(w): sentence IDs containing word w."""
        return self.graph.sentences_containing_word(w)

    # --- Next-word counts and transition P ---

    def N_w(self, w: int) -> dict[int, int]:
        """N_w(w): next-word counts from w (word_next)."""
        return self.graph.next_word_counts(w)

    def P(self, w: int) -> dict[int, float]:
        """
        P(·|w): row-stochastic distribution over next word IDs from w.
        Normalized from N_w(w). Returns empty dict if no successors.
        """
        counts = self.graph.next_word_counts(w)
        if not counts:
            return {}
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {w_next: c / total for w_next, c in counts.items()}

    # --- Co-occurrence and sentence similarity ---

    def Co(self, w: int) -> list[int]:
        """Co(w): co-occurring word IDs (word_cooccurrence)."""
        return self.graph.cooccurring_words(w)

    def J(self, s: int, top_k: int = 10) -> list[tuple[int, float]]:
        """J(s, s'): similar sentences to s as (sentence_id, jaccard) pairs."""
        return self.graph.similar_sentences(s, top_k=top_k)

    # --- Pipeline: Embed_anchor, Propagation_layer, Output_head, Refine ---

    def Embed_anchor(
        self,
        query_token_ids: list[int] | None = None,
        use_definition_words: bool = False,
        definition_word_weight: float = 0.5,
        active_categories: set[int] | None = None,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        Embed_anchor(q, C(q), x_q, ...) → (v_W_0, v_S_0).
        Builds initial state from concept_bundle terms and optional query_token_ids.
        """
        return graph_attention.embed_anchor(
            self.concept_bundle,
            self.graph,
            self.word_to_id,
            query_token_ids=query_token_ids,
            use_definition_words=use_definition_words,
            definition_word_weight=definition_word_weight,
            active_categories=active_categories,
        )

    def Propagation_layer(
        self,
        v_W: dict[int, float],
        v_S: dict[int, float],
        genre_id: str | list[str] | None,
        use_weights: bool = True,
        use_cooccurrence: bool = False,
        use_backward: bool = False,
        content_dependent_j: bool = False,
        overlay: dict[str, Any] | None = None,
        active_categories: set[int] | None = None,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        One hop of propagation: word→sentence, sentence→sentence (J), sentence→word, word→word (P).
        """
        return graph_attention.propagation_layer(
            v_W,
            v_S,
            self.graph,
            genre_id,
            self.encoded_index,
            use_weights,
            use_cooccurrence=use_cooccurrence,
            use_backward=use_backward,
            content_dependent_j=content_dependent_j,
            overlay=overlay,
            active_categories=active_categories,
        )

    def Output_head(
        self,
        v_W: dict[int, float],
        dict_term_ids: set[int] | None = None,
        dict_boost: float = 0.0,
    ) -> dict[int, float]:
        """
        Output head: optionally boost dictionary terms, then L1-normalize v_W to distribution p.
        """
        return graph_attention.output_head(
            v_W,
            dict_term_ids=dict_term_ids,
            dict_boost=dict_boost,
        )

    def Refine(
        self,
        pattern_word_ids: list[int],
        pattern_sentence_ids: list[int],
        genre_id: str | list[str] = "general",
        max_sentences: int = 15,
        max_definitions: int = 10,
        secondary_word_ids: list[int] | None = None,
        secondary_sentence_ids: list[int] | None = None,
        next_span_sentence_ids: list[int] | None = None,
        sentence_visits: dict[int, float] | None = None,
        output_format: str = "list",
        paragraph_max_chars: int = 500,
        include_definitions: bool = True,
        **kwargs: Any,
    ) -> str | tuple[str, list[dict[str, Any]]]:
        """
        Refine: map pattern (top words + sentences) to response text using
        concept_bundle definitions and encoded_index sentence texts.
        """
        return graph_attention.refine_answer(
            pattern_word_ids,
            pattern_sentence_ids,
            self.concept_bundle,
            self.encoded_index,
            self.id_to_word,
            genre_id=genre_id,
            max_sentences=max_sentences,
            max_definitions=max_definitions,
            secondary_word_ids=secondary_word_ids,
            secondary_sentence_ids=secondary_sentence_ids,
            next_span_sentence_ids=next_span_sentence_ids,
            sentence_visits=sentence_visits,
            output_format=output_format,
            paragraph_max_chars=paragraph_max_chars,
            include_definitions=include_definitions,
            **kwargs,
        )
