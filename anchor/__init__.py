"""
Anchor: dictionary as symbolic anchor for LM output.
Keeps generation honest (on-concept, verifiable); responses can be any genre.
"""
from .engine import AnchorEngine
from .critic import dictionary_score, score_and_decide
from .retrieval import get_concept_bundle, get_style_sentences
from .generator import generate
from .wire import get_engine, get_config, get_generator_kind

__all__ = [
    "AnchorEngine",
    "dictionary_score",
    "score_and_decide",
    "get_concept_bundle",
    "get_style_sentences",
    "generate",
    "get_engine",
    "get_config",
    "get_generator_kind",
]
