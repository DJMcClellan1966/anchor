# Anchor

Anchor uses the **dictionary as a symbolic anchor** so an LM stays honest about what it talks about; responses can be any genre (definitional, narrative, etc.) via style retrieval and optional LM.

## What it does

- **Concept bundle** from the dictionary (BasisEngine) for a query.
- **Style sentences** from genre corpus (optional) filtered by concept.
- **Generate** via stub (terms + definitions), optionally scratchLLM/Align, **corpus** (Option C: graph-based next-sentence or hybrid next-token), or **graph attention** (query lights up the graph, attention loops, pattern-based refinement; grounded, non-hallucinatory).
- **Critic** scores the response against the graph (accept/warn/reject).

### Graph LLM (new type of LLM)

When the graph exists and `use_attention_loop` is true, Anchor runs as a **graph-grounded LLM**: a distinct type of language model (CPU-only, no hallucination by design). The same pipeline is the **Graph LLM forward pass**:

1. **Input:** query.
2. **Concept bundle:** dictionary lookup for terms and definitions.
3. **Activate:** query terms map to word/sentence nodes in the graph.
4. **Traverse:** attention-like propagation (relationship weights, next-word, multi-hop).
5. **Pattern:** primary + secondary path groups and next-span sentences.
6. **Refine:** definitions + ordered sentences → response (optionally one paragraph). By default responses are **LLM-like** (single coherent answer); set `include_definitions_in_response: true` to include definition lines.
7. **Critic:** accept/warn/reject against the dictionary.

Optional add-ons (on by default): **vector geometry** (lightweight word vectors from the graph via `scripts/build_vectors.py`; `use_graph_vectors` is true so when `word_vectors.json` exists, sentence visits are boosted by query–sentence similarity) and **feedback-driven adaptation**: run `python run_anchor.py "Your question?" --feedback accept` to log accepted responses; then `python scripts/build_feedback_weights.py data` to build `feedback_weights.json`; on later runs, sentence_visits are boosted for sentences that appeared in accepted responses to the same query. See Config and Project layout below.

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

### ScratchLLM

To use ScratchLLM for generation (fluent, parametric output with dictionary and style context):

1. Set **config/paths.json** `scratchllm_path` to the ScratchLLM repo root (directory that contains `base/retrieve.py`).
2. Set **config/anchor.json** `use_scratchllm` to `true` to prefer ScratchLLM over the graph generator when both are available.
3. ScratchLLM must expose `from base.retrieve import retrieve_formal_only` and `retrieve_formal_only(question, truth_base_path=<path_to_jsonl>, top_k=5)` returning a string. Anchor builds the truth base from concept definitions (tier 1, source "dictionary") and style sentences (tier 2, source "genre") and passes it as a temp file.

## Run

From the anchor folder:

```bash
python run_anchor.py "What is a function?"
```

Use `-v` or `--verbose` to show the pipeline (generator kind, concepts, style sentence count, critic grounded/total). Use `--stream` to stream the response in chunks (LLM-like). Use `--feedback accept` or `--feedback reject` to record feedback for the response (for adaptation; see below). In REPL mode, recent turns are used as context (see `conversation_turn_limit`).

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
| `use_scratchllm` | `false` | When true and `scratchllm_path` set, use ScratchLLM for generation (context = definitions + style sentences). Overrides graph generator when both available. |
| `include_definitions_in_response` | `false` | When false (default), responses are answer-style only (terms + sentences/paragraph). When true, stub and graph generator include definition lines. |
| `system_prompt` | `null` | Optional. When set, prepended to query for concept lookup and to ScratchLLM truth base (role/instruction). |
| `conversation_turn_limit` | `2` | In REPL, number of prior (Q, A) pairs merged into context for the next query. |
| `streaming_max_chunk_chars` | `120` | When using `--stream`, max chunk size when no sentence boundary. |
| `use_attention_loop` | `true` | When true (and graph exists), use graph-attention generator: query activates nodes, attention traverses loops (with optional relationship weights and next-word propagation), multiple path groups and next-span sentences are combined into the answer (grounded, non-hallucinatory). Set to `false` to use corpus (hybrid next-token) instead. |
| `use_style_sentences` | `true` | Style sentences from corpus or per-genre files |
| `use_critic` | `true` | Dictionary-based grounding score and accept/warn/reject |

## Config

- **config/paths.json** – `dictionary_path`, `scratchllm_path`, `align_data_dir`
- **config/anchor.json** – `use_dictionary`, `use_corpus_graph`, `use_scratchllm`, `include_definitions_in_response`, `system_prompt`, `conversation_turn_limit`, `streaming_max_chunk_chars`, `use_attention_loop`, `attention_loop_hops` (default 4), `attention_loop_top_k`, `attention_loop_use_weights`, `attention_loop_path_groups`, `attention_loop_next_span`, `attention_loop_max_iter`, `attention_loop_output_format` (default paragraph), `attention_loop_paragraph_max_chars`, `use_graph_vectors` (default true), `graph_vectors_boost`, `feedback_path`, `feedback_weights_path`, `feedback_boost`, `use_style_sentences`, `use_critic`, `critic_accept_threshold`, `critic_low_warn_threshold`, `default_genre_id`, `register`, `next_sentence_mode`, `corpus_next_sentences_top_k`, `corpus_hybrid_context_length`, `corpus_hybrid_beta`, `corpus_max_tokens`

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
- **anchor/graph_attention.py** – Graph attention loop: activate from query, traverse with optional edge weights and next-word propagation, detect pattern (primary + secondary path groups), optional next-span from similar sentences, refine answer (grounded)
- **anchor/corpus_model.py** – Load and sample from by-product corpus model (1-layer LM)
- **anchor/wire.py** – Load config and dictionary (single place that touches external repos)
- **run_anchor.py** – CLI entry point
- **scripts/build_corpus.py** – Build combined corpus with genre tags (from local files)
- **scripts/build_corpus_from_hf.py** – Build corpus from Hugging Face datasets (OpenSubtitles, C4, etc.; requires `datasets`)
- **scripts/build_vocab.py** – Build vocab and encoded_sentences from corpus
- **scripts/build_graph.py** – Build corpus graph (with inverted index) and optional corpus_model.json from encoded sentences
- **scripts/build_vectors.py** – Build word vectors from graph for optional vector geometry (requires graph.json; optional scipy for SVD)
- **anchor/graph_vectors.py** – Load word vectors and boost sentence visits by query–sentence similarity when use_graph_vectors is true
- **anchor/feedback.py** – Record accept/reject feedback; load feedback_weights and boost sentence visits for adaptation
- **scripts/build_feedback_weights.py** – Build feedback_weights.json from feedback.jsonl (accepted responses) for Graph LLM adaptation
