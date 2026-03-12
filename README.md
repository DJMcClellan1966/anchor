# Anchor

Anchor uses the **dictionary as a symbolic anchor** so an LM stays honest about what it talks about; responses can be any genre (definitional, narrative, etc.) via style retrieval and optional LM.

## What it does

- **Concept bundle** from the dictionary (BasisEngine) for a query.
- **Style sentences** from genre corpus (optional) filtered by concept.
- **Generate** via stub (terms + definitions), optionally scratchLLM/Align, **corpus** (Option C: graph-based next-sentence or hybrid next-token), or **graph attention** (query lights up the graph, attention loops, pattern-based refinement; grounded, non-hallucinatory).
- **Critic** scores the response against the graph (accept/warn/reject).

## Option C: Combined corpus with genre tags

You can build a single sentence corpus, vocabulary, and word/sentence graph for retrieval and next-sentence prediction (all local, CPU-friendly):

1. **Build corpus** — Put source text in a directory (e.g. `.txt` or `.jsonl`), then:
   ```bash
   python scripts/build_corpus.py path/to/sources -o data
   ```
   Writes `data/corpus/sentences.jsonl` (and optionally `data/<genre_id>/genre_sentences.jsonl`).

   **From Hugging Face:** Install optional deps: `pip install -r requirements-corpus.txt` (or `pip install datasets`). Then build from C4, Common Corpus, or similar (use `--streaming` for large sets like C4):
   ```bash
   python scripts/build_corpus_from_hf.py --dataset allenai/c4 --config en -o data --streaming --max-sentences 100000
   python scripts/build_corpus_from_hf.py --dataset PleIAs/common_corpus -o data --max-sentences 50000
   ```
   If the dataset uses a different text column, add `--text-field <name>`. Then run the vocab and graph steps below.

2. **Build vocabulary and encode** — Run after the corpus step:
   ```bash
   python scripts/build_vocab.py data
   ```
   Writes `data/corpus/vocab.json` and `data/corpus/encoded_sentences.jsonl`.

3. **Build graph** — Run after the vocab step:
   ```bash
   python scripts/build_graph.py data
   ```
   Writes `data/corpus/graph.json` (with inverted index for context→sentences) and, if vocab exists, `data/corpus/corpus_model.json` (by-product transition matrix for a 1-layer LM). Use `--no-corpus-model` to skip the corpus model. Use `--context-length 5` (default) for the context window.

4. **Use in Anchor** — Set `align_data_dir` in config/paths.json to `data` (or the directory containing `corpus/`). Option C is **on by default** when the graph exists; set `"use_corpus_graph": false` in config/anchor.json to turn it off.

## Tests

Install test deps: `pip install -r requirements-test.txt`

Run tests (excludes slow real-engine test by default):

```bash
pytest tests/ -m "not real_engine"
```

Run including the real dictionary engine (when `dictionary_path` is set):

```bash
pytest tests/ -m real_engine
```

Tests cover errors, completeness, robustness, and breakage (invalid input, missing files, malformed JSON). Integration tests run the full corpus pipeline and engine with a mock.

## Prerequisites

- Python 3.10+
- [Dictionary](https://github.com/DJMcClellan1966/dictionary) repo (for BasisEngine and concept graph)
- Optional: scratchLLM or Align for richer generation

## Setup

1. Clone or place the **dictionary** repo (e.g. `Desktop/dictionary`).
2. Edit **config/paths.json**: set `dictionary_path` to the dictionary repo root (e.g. `.../dictionary/dictionary`). Copy from **config/paths.json.example** if needed.
3. Optional: set `scratchllm_path` or `align_data_dir` to use scratchLLM or Align for generation.

## Run

From the anchor folder:

```bash
python run_anchor.py "What is a function?"
```

Or start a REPL:

```bash
python run_anchor.py --repl
```

If `dictionary_path` is not set, Anchor prints a message and exits. Set it in **config/paths.json** or via **ANCHOR_DICTIONARY_PATH**.

## Features (all on by default)

When the required paths/data exist, these are used automatically. Set to `false` in **config/anchor.json** to turn off:

| Config key | Default | When used |
|------------|---------|-----------|
| `use_dictionary` | `true` | Dictionary at `dictionary_path` for concept bundle and critic |
| `use_corpus_graph` | `true` | Graph at `align_data_dir/corpus/graph.json` for style retrieval and corpus generator |
| `use_attention_loop` | `true` | When true (and graph exists), use graph-attention generator: query activates nodes, attention traverses loops, repeating pattern refines the answer (grounded, non-hallucinatory). Set to `false` to use corpus (hybrid next-token) instead. |
| `use_style_sentences` | `true` | Style sentences from corpus or per-genre files |
| `use_critic` | `true` | Dictionary-based grounding score and accept/warn/reject |

## Config

- **config/paths.json** – `dictionary_path`, `scratchllm_path`, `align_data_dir`
- **config/anchor.json** – `use_dictionary`, `use_corpus_graph`, `use_attention_loop`, `attention_loop_hops`, `attention_loop_top_k`, `use_style_sentences`, `use_critic`, `critic_accept_threshold`, `critic_low_warn_threshold`, `default_genre_id`, `register`, `next_sentence_mode`, `corpus_next_sentences_top_k`, `corpus_hybrid_context_length`, `corpus_hybrid_beta`, `corpus_max_tokens`

Env overrides: `ANCHOR_DICTIONARY_PATH`, `ANCHOR_DATA_DIR`.

## Project layout

- **anchor/engine.py** – AnchorEngine: concept -> style -> generate -> critic
- **anchor/critic.py** – Dictionary score and accept/warn/reject
- **anchor/retrieval.py** – Concept bundle, style sentences, and graph-based retrieval (Option C)
- **anchor/generator.py** – Stub, optional scratchLLM/Align, or corpus (next-sentence or hybrid next-token)
- **anchor/corpus_vocab.py** – Vocabulary build and sentence encoding (Option C)
- **anchor/corpus_graph.py** – Word/sentence graph build and load; by-product transition matrix (Option C)
- **anchor/next_sentence.py** – Next-sentence retrieval from graph (Option C)
- **anchor/next_token.py** – Retrieval + bigram hybrid next-token distribution and sampling
- **anchor/graph_attention.py** – Graph attention loop: activate from query, traverse with attention, detect pattern, refine answer (grounded)
- **anchor/corpus_model.py** – Load and sample from by-product corpus model (1-layer LM)
- **anchor/wire.py** – Load config and dictionary (single place that touches external repos)
- **run_anchor.py** – CLI entry point
- **scripts/build_corpus.py** – Build combined corpus with genre tags (from local files)
- **scripts/build_corpus_from_hf.py** – Build corpus from Hugging Face datasets (OpenSubtitles, C4, etc.; requires `datasets`)
- **scripts/build_vocab.py** – Build vocab and encoded_sentences from corpus
- **scripts/build_graph.py** – Build corpus graph (with inverted index) and optional corpus_model.json from encoded sentences
