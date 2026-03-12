"""
Build lightweight word vectors from the corpus graph (Option C).
Run after build_graph.py. Uses co-occurrence counts and optional SVD for low-dim vectors.
Output: corpus/word_vectors.json for use with use_graph_vectors in Graph LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))


def _build_cooccurrence_from_graph(graph: dict) -> tuple[list[int], list[tuple[int, int, float]]]:
    """From graph sentence_words build (vocab_list, [(row_idx, col_idx, count), ...])."""
    sentence_words = graph.get("sentence_words", {})
    all_wids: set[int] = set()
    for sid, tids in sentence_words.items():
        for w in tids:
            all_wids.add(int(w))
    vocab = sorted(all_wids)
    wid_to_idx = {w: i for i, w in enumerate(vocab)}
    rows, cols, data = [], [], []
    for sid, tids in sentence_words.items():
        token_ids = [int(t) for t in tids]
        for i, wi in enumerate(token_ids):
            for j, wj in enumerate(token_ids):
                if wi == wj:
                    continue
                ri, ci = wid_to_idx[wi], wid_to_idx[wj]
                rows.append(ri)
                cols.append(ci)
                data.append(1.0)
    return vocab, list(zip(rows, cols, data))


def build_word_vectors(
    graph_path: Path,
    output_path: Path,
    dim: int = 50,
) -> None:
    """
    Build word vectors from graph co-occurrence and write word_vectors.json.
    Uses SVD if scipy is available; otherwise uses top-dim raw co-occurrence rows.
    """
    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)
    vocab, triples = _build_cooccurrence_from_graph(graph)
    if not vocab or not triples:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dim": 0, "word_vectors": {}}
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False)
        return

    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import svds
    except ImportError:
        # No SVD: use first dim co-occurrence dimensions as proxy (no reduction)
        # Build dense rows for each word from triples
        n = len(vocab)
        from collections import defaultdict
        row_data: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for ri, ci, v in triples:
            row_data[ri][ci] = row_data[ri].get(ci, 0) + v
        # Take first `dim` columns (by vocab order) as vector
        dim_actual = min(dim, n)
        word_vectors = {}
        for i, wid in enumerate(vocab):
            row = row_data.get(i, {})
            vec = [float(row.get(j, 0)) for j in range(dim_actual)]
            # L2 normalize
            norm = (sum(x * x for x in vec)) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            word_vectors[str(wid)] = vec
        payload = {"dim": dim_actual, "word_vectors": word_vectors}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False)
        return

    n = len(vocab)
    row_idx = [t[0] for t in triples]
    col_idx = [t[1] for t in triples]
    data_vals = [t[2] for t in triples]
    M = sp.coo_matrix((data_vals, (row_idx, col_idx)), shape=(n, n))
    M = M.tocsr()
    k = min(dim, n - 1, M.nnz // max(n, 1))
    if k < 1:
        k = 1
    try:
        U, _s, _Vt = svds(M.astype(float), k=k)
    except Exception:
        U = M.toarray()[:, :dim] if n >= dim else M.toarray()
        k = U.shape[1]
    if U.shape[1] != k and U.shape[1] > 0:
        k = U.shape[1]
    word_vectors = {}
    for i, wid in enumerate(vocab):
        vec = [float(U[i, j]) for j in range(k)]
        norm = (sum(x * x for x in vec)) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        word_vectors[str(wid)] = vec
    payload = {"dim": k, "word_vectors": word_vectors}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build word vectors from corpus graph for Graph LLM vector geometry."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Data directory containing corpus/graph.json (default: data)",
    )
    parser.add_argument(
        "--graph-file",
        default="corpus/graph.json",
        help="Path to graph relative to data_dir (default: corpus/graph.json)",
    )
    parser.add_argument(
        "--output",
        default="corpus/word_vectors.json",
        help="Output path relative to data_dir (default: corpus/word_vectors.json)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=50,
        help="Vector dimension for SVD (default: 50)",
    )
    args = parser.parse_args()

    graph_path = args.data_dir / args.graph_file
    if not graph_path.exists():
        print(f"Missing {graph_path}. Run build_graph.py first.")
        raise SystemExit(1)

    output_path = args.data_dir / args.output
    build_word_vectors(graph_path, output_path, dim=args.dim)
    print(f"Wrote word vectors to {output_path}")


if __name__ == "__main__":
    main()
