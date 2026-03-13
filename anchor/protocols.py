"""
Protocols (traits) for Anchor pipeline: Vocabulary, Graph, Propagation, Critic.

Existing implementations: CorpusGraph (GraphLike), (word_to_id, id_to_word) + tokenize (VocabularyLike),
run_layers (PropagationLike), critic.score_and_decide (CriticLike).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VocabularyLike(Protocol):
    """Vocabulary: word string <-> word ID. Tokenize is typically provided separately (e.g. corpus_vocab.tokenize)."""

    word_to_id: dict[str, int]
    id_to_word: dict[int, str]


@runtime_checkable
class GraphLike(Protocol):
    """Graph: sentences, words, S(w), W(s), P, J, Co. Optional category methods."""

    def sentence_ids(self) -> list[int]:
        """All sentence IDs."""
        ...

    def sentence_token_ids(self, sentence_id: int) -> list[int]:
        """Token ID sequence for sentence."""
        ...

    def sentences_containing_word(self, word_id: int) -> list[int]:
        """Sentence IDs containing this word."""
        ...

    def next_word_counts(self, word_id: int) -> dict[int, int]:
        """Next-word counts from this word."""
        ...

    def similar_sentences(self, sentence_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """(sentence_id, jaccard) pairs for similar sentences."""
        ...

    def cooccurring_words(self, word_id: int) -> list[int]:
        """Word IDs that co-occur with this word."""
        ...

    def has_category_data(self) -> bool:
        """True if graph has word-to-category data (optional)."""
        ...

    def word_category(self, word_id: int) -> int | None:
        """Category ID for word, or None (optional)."""
        ...

    def sentences_containing_word_in_categories(
        self, word_id: int, category_set: set[int]
    ) -> list[int]:
        """Sentence IDs containing word in given categories (optional)."""
        ...


@runtime_checkable
class CriticLike(Protocol):
    """Critic: score response and decide accept/warn/reject."""

    def score_and_decide(
        self,
        response_text: str,
        engine: Any,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return dict with score, num_grounded, num_content, decision, show_warning, message."""
        ...
