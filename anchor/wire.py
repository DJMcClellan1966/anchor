"""
Wire: load config, sys.path for dictionary (and optionally scratchLLM), return engine and opts.
Single place that touches external repos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ANCHOR_ROOT = Path(__file__).resolve().parent.parent


def _load_paths() -> dict[str, Any]:
    p = _ANCHOR_ROOT / "config" / "paths.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _load_anchor_config() -> dict[str, Any]:
    p = _ANCHOR_ROOT / "config" / "anchor.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


_RELATIVE_PATH_KEYS = (
    "align_data_dir",
    "feedback_path",
    "feedback_weights_path",
    "grammar_rules_path",
    "grammar_examples_path",
)


def get_config() -> dict[str, Any]:
    """Merged config: paths + anchor.json. Env overrides: ANCHOR_DICTIONARY_PATH, etc.
    Relative paths for align_data_dir, feedback_path, etc. are resolved against project root."""
    paths = _load_paths()
    anchor_cfg = _load_anchor_config()
    out = {**paths, **anchor_cfg}
    if os.environ.get("ANCHOR_DICTIONARY_PATH"):
        out["dictionary_path"] = os.environ["ANCHOR_DICTIONARY_PATH"]
    if os.environ.get("ANCHOR_DATA_DIR"):
        out["align_data_dir"] = os.environ["ANCHOR_DATA_DIR"]
    for key in _RELATIVE_PATH_KEYS:
        val = out.get(key)
        if isinstance(val, str) and val.strip() and not Path(val).is_absolute():
            out[key] = str((_ANCHOR_ROOT / val).resolve())
    return out


def get_engine() -> Any | None:
    """Build dictionary engine. Prefers Webster if webster_json_path set and exists; else BasisEngine from dictionary_path."""
    cfg = get_config()
    if not cfg.get("use_dictionary", True):
        return None
    webster_path = cfg.get("webster_json_path")
    if webster_path:
        path = Path(webster_path).resolve()
        if path.exists():
            try:
                from anchor.webster_engine import WebsterEngine
                return WebsterEngine(path)
            except ImportError:
                from webster_engine import WebsterEngine
                return WebsterEngine(path)
    dict_path = cfg.get("dictionary_path")
    if not dict_path:
        return None
    path = Path(dict_path).resolve()
    if not path.exists():
        return None
    import sys
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        from src.basis_engine import BasisEngine
        return BasisEngine(project_root=path)
    except ImportError:
        return None


def get_generator_kind() -> str:
    """Return 'stub' | 'graph_attention' | 'corpus' | 'scratchllm' | 'align' based on config and paths."""
    cfg = get_config()
    if cfg.get("use_scratchllm", False) and cfg.get("scratchllm_path") and Path(cfg["scratchllm_path"]).exists():
        return "scratchllm"
    if cfg.get("use_graph_llm", True) and cfg.get("use_corpus_graph", True) and cfg.get("align_data_dir"):
        data_path = Path(cfg["align_data_dir"])
        if data_path.exists() and (data_path / "corpus" / "graph.json").exists():
            if cfg.get("use_attention_loop", True):
                return "graph_attention"
            return "corpus"
    if cfg.get("align_data_dir") and Path(cfg["align_data_dir"]).exists():
        align_root = _ANCHOR_ROOT.parent / "align"
        if align_root.exists() and (align_root / "Align" / "respond_bridge.py").exists():
            return "align"
    if cfg.get("scratchllm_path") and Path(cfg["scratchllm_path"]).exists():
        return "scratchllm"
    return "stub"
