# Anchor

Anchor uses the **dictionary as a symbolic anchor** so an LM stays honest about what it talks about; responses can be any genre (definitional, narrative, etc.) via style retrieval and optional LM.

## What it does

- **Concept bundle** from the dictionary (BasisEngine or Webster JSON) for a query.
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

Graph LLM mode uses: `use_corpus_graph`, `use_attention_loop`, `use_query_token_ids`, `use_naturalize`, `use_graph_vectors` (all on by default). Set `use_graph_llm` to `false` to turn off the graph path entirely.

Optional add-ons (on by default): **vector geometry** (lightweight word vectors from the graph via `scripts/build_vectors.py`; `use_graph_vectors` is true so when `word_vectors.json` exists, sentence visits are boosted by query–sentence similarity) and **feedback-driven adaptation**: run `python run_anchor.py "Your question?" --feedback accept` to log accepted responses; then `python scripts/build_feedback_weights.py data` to build `feedback_weights.json`; on later runs, sentence_visits are boosted for sentences that appeared in accepted responses to the same query. See Config and Project layout below.

## Option C: Combined corpus with genre tags

You can build a single sentence corpus, vocabulary, and word/sentence graph for retrieval and next-sentence prediction (all local, CPU-friendly):

1. **Build corpus** — Put source text in a directory (e.g. `.txt` or `.jsonl`), then:
   ```bash
   python scripts/build_corpus.py path/to/sources -o data
   ```
   Use `--genre encyclopedia` (or any genre_id) to force that genre for every sentence. Writes `data/corpus/sentences.jsonl` (and optionally `data/<genre_id>/genre_sentences.jsonl`).

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

### Encyclopedia (world knowledge)

Add encyclopedic content as a corpus genre so answers can use factual, encyclopedic sentences. Put encyclopedia text in a directory and run:

```bash
python scripts/build_corpus.py path/to/encyclopedia -o data --genre encyclopedia
```

To merge with an existing corpus, run the above with the same `-o data` (this overwrites `sentences.jsonl`; to merge, append encyclopedic sentences to `data/corpus/sentences.jsonl` with `"genre_id": "encyclopedia"` then re-run vocab and graph build). Set `default_genre_id` to `"encyclopedia"` in config to prefer encyclopedic answers, or set `genre_ids` to `["retirement", "encyclopedia"]` (or similar) in config to mix style and world knowledge.

### Webster dictionary (optional backend + definitional corpus)

Use [Webster's English Dictionary (JSON)](https://github.com/matthewreagan/WebstersEnglishDictionary) as **both** the dictionary engine and definitional corpus so concept bundles, critic, and graph retrieval all use the same source.

1. **Set Webster as dictionary** — In **config/paths.json** set `webster_json_path` to your `dictionary.json` or `dictionary_compact.json` (e.g. from a clone of WebstersEnglishDictionary). Set `align_data_dir` to `"data"` (or your corpus directory) so the graph is used.
2. **Add Webster to the corpus** — Run once to append definitional sentences to your existing corpus (then rebuild vocab and graph):
   ```bash
   python scripts/build_corpus_from_webster.py path/to/dictionary.json -o data --genre definitional --append
   python scripts/build_vocab.py data
   python scripts/build_graph.py data
   ```
   Each line is written with `source: "dictionary"` and `term: "<term>"` so definitions live in the same sequence store as the corpus (unified data store).
3. **Use definitional content in retrieval** — In **config/anchor.json** set `genre_ids` to include `"definitional"` (e.g. `["retirement", "definitional"]`). Default is both so dictionary and corpus both use Webster.

**Unified data store:** When dictionary is ingested via `build_corpus_from_webster.py`, definition rows are first-class sentences in `sentences.jsonl` (and thus in `encoded_sentences.jsonl` and the graph). Build the graph from that single encoded file only (do not pass `encoded_dictionary.jsonl`; use `--no-encoded-dictionary` if you previously merged a separate encoded dictionary). Refinement then prefers definition text from the encoded index (rows with `term`) over the concept bundle, so dictionary and corpus data live in the same place.

### Dictionary repo corpus (ConceptNet, GooAQ, 3M+)

The [dictionary repo](https://github.com/DJMcClellan1966/dictionary) can compile large corpora (ConceptNet ~34M assertions filtered to English, **GooAQ 3M+ Q&A pairs**, The Stack docstrings, Python stdlib). To use that same data in Anchor:

1. **Build the dictionary’s compiled corpora** (one-time, in the dictionary repo):
   ```bash
   cd path/to/dictionary
   python compile_corpus.py --source gooaq    # ~800MB download, then compile
   python compile_corpus.py --source conceptnet
   # optional: --source stack, --source stdlib; use --max 500000 for testing
   ```
2. **Ingest into Anchor’s corpus** — From the Anchor project:
   ```bash
   python scripts/build_corpus_from_dictionary.py path/to/dictionary/data -o data --append
   python scripts/build_vocab.py data
   python scripts/build_graph.py data
   ```
   Use `--sources compiled_corpus.json compiled_conceptnet.json` to pick which files; use `--max N` to cap sentences per source. Output lines have `genre_id` set to `gooaq`, `conceptnet`, `stack`, or `stdlib`. Add those to `genre_ids` in **config/anchor.json** if you want retrieval to use them (e.g. `["retirement", "definitional", "gooaq", "conceptnet"]`).

### Dictionary as numbers and natural language

Encode the dictionary as token ID sequences (same vocab as the corpus) and merge them into the graph so recurring patterns (word_next) include definition text. Optionally extend responses for more natural phrasing.

1. **Build corpus and vocab** (as usual), then **encode the dictionary**:
   ```bash
   python scripts/build_vocab.py data
   python scripts/encode_dictionary.py --vocab data/corpus/vocab.json --webster path/to/dictionary.json --output data/corpus/encoded_dictionary.jsonl
   ```
   Or use `--definitions path/to/terms_defs.txt` (format: `term\tdefinition` per line) instead of `--webster`.
2. **Rebuild the graph** — Dictionary-as-numbers merge is **automatic**: if `data/corpus/encoded_dictionary.jsonl` exists, it is merged into the graph. Use `--no-encoded-dictionary` to skip merging (opt out).
   ```bash
   python scripts/build_graph.py data
   ```
3. **Naturalize (on by default)** — Response tails are extended using the graph’s next-token patterns when the graph exists. Set `use_naturalize: false` in **config/anchor.json** to turn this off. Config: `naturalize_max_tokens` (default 12), `naturalize_context_length` (default 5).

### Grammar (fluency)

Optional grammar rewrite runs after generation. Set `use_grammar: true` and either `grammar_rules_path` (JSON array of `{"pattern": "regex", "replacement": "string"}`) or `grammar_command` (shell command reading stdin, writing corrected text to stdout). Grammar examples for ScratchLLM: set `grammar_examples_path` to a file (one sentence or JSONL with `text` per line); when using ScratchLLM, those lines are appended to the truth base (source "grammar").

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
- Dictionary: either the [Dictionary](https://github.com/DJMcClellan1966/dictionary) repo (BasisEngine) or a [Webster JSON](https://github.com/matthewreagan/WebstersEnglishDictionary) file (`webster_json_path`)
- Optional: scratchLLM or Align for richer generation

## Setup

1. **Dictionary:** Either (a) clone the **dictionary** repo and set `dictionary_path` in config/paths.json to its root, or (b) set `webster_json_path` to a Webster JSON file (e.g. from [WebstersEnglishDictionary](https://github.com/matthewreagan/WebstersEnglishDictionary) — `dictionary.json` or `dictionary_compact.json`). If both are set, Webster is used.
2. Edit **config/paths.json**: `dictionary_path`, optional `webster_json_path`, optional `scratchllm_path`, `align_data_dir`. Copy from **config/paths.json.example** if needed.
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

Use `-v` or `--verbose` to show the pipeline (generator kind, concepts, style sentence count, critic grounded/total). Use `--stream` to stream the response in chunks (LLM-like). Use `--feedback accept` or `--feedback reject` to record feedback for the response (for adaptation; see below). In REPL mode, recent turns are used as context (see `conversation_turn_limit`). Run `python run_anchor.py --check` to verify setup before querying.

Or start the GUI (question box and response area):

```bash
python run_anchor_gui.py
```

Or start a REPL:

```bash
python run_anchor.py --repl
```

If neither `dictionary_path` nor `webster_json_path` is set (and valid), Anchor prints a message and exits. Set one in **config/paths.json** or use **ANCHOR_DICTIONARY_PATH** for the Basis path.

### Minimal run

Shortest path to a successful graph run:

1. **Set dictionary** — In **config/paths.json** set `dictionary_path` (or `webster_json_path`).
2. **Build corpus and graph** — Run `build_corpus` on a small folder or file, then `build_vocab`, then `build_graph` (all with `-o data` or your directory).
3. **Set data dir** — In **config/paths.json** set `align_data_dir` to that directory (e.g. `"data"`).
4. **Run** — `python run_anchor.py "What is X?"`

You should see `Generator: graph_attention (graph: N sentences, vocab: M)` and a response that uses your corpus. Run `python run_anchor.py --check` to verify setup before querying.

## Features (all on by default)

When the required paths/data exist, these are used automatically. Set to `false` in **config/anchor.json** to turn off:

| Config key | Default | When used |
|------------|---------|-----------|
| `use_dictionary` | `true` | Dictionary at `dictionary_path` or Webster at `webster_json_path` for concept bundle and critic |
| `use_graph_llm` | `true` | When true (default), graph-based generation (graph_attention or corpus) is used when data exists. Set to false to force stub and disable the graph path. |
| `use_corpus_graph` | `true` | Graph at `align_data_dir/corpus/graph.json` for style retrieval and corpus generator |
| `use_scratchllm` | `false` | When true and `scratchllm_path` set, use ScratchLLM for generation (context = definitions + style sentences). Overrides graph generator when both available. |
| `include_definitions_in_response` | `false` | When false (default), responses are answer-style only (terms + sentences/paragraph). When true, stub and graph generator include definition lines. |
| `system_prompt` | `null` | Optional. When set, prepended to query for concept lookup and to ScratchLLM truth base (role/instruction). |
| `conversation_turn_limit` | `2` | In REPL, number of prior (Q, A) pairs merged into context for the next query. |
| `streaming_max_chunk_chars` | `120` | When using `--stream`, max chunk size when no sentence boundary. |
| `genre_ids` | `null` | When set to an array (e.g. `["retirement", "encyclopedia"]`), retrieval and graph use sentences from any of these genres. When null, only `default_genre_id` is used. |
| `use_grammar` | `false` | When true, run grammar rewrite after generation (requires `grammar_rules_path` or `grammar_command`). |
| `grammar_rules_path` | `null` | Path to JSON array of `{"pattern": "regex", "replacement": "string"}` applied in order. |
| `grammar_command` | `null` | Shell command that reads stdin and writes corrected text to stdout (used when set, instead of rules file). |
| `grammar_examples_path` | `null` | Path to file of example sentences (or JSONL with `text`). When set and using ScratchLLM, appended to truth base as source "grammar". |
| `use_naturalize` | `true` | When true (default), extend response tail using graph next-token patterns; set to `false` to turn off. |
| `naturalize_max_tokens` | `12` | Max tokens to append in naturalize step. |
| `naturalize_context_length` | `5` | Context length (tokens) for next-token sampling in naturalize. |
| `use_query_token_ids` | `true` | When true (default), query is tokenized with corpus vocab and token IDs are used for graph activation in addition to dictionary terms; set to `false` to activate from dictionary terms only. |
| `use_attention_loop` | `true` | When true (and graph exists), use graph-attention generator: query activates nodes, attention traverses loops (with optional relationship weights and next-word propagation), multiple path groups and next-span sentences are combined into the answer (grounded, non-hallucinatory). Set to `false` to use corpus (hybrid next-token) instead. |
| `use_style_sentences` | `true` | Style sentences from corpus or per-genre files |
| `use_critic` | `true` | Dictionary-based grounding score and accept/warn/reject |
| `use_propagation_cooccurrence` | `true` | When true, propagation spreads activation via co-occurrence edges (Co(w)) |
| `use_propagation_backward` | `true` | When true, add backward word→word step (P_prev) in propagation |
| `use_content_dependent_j` | `true` | When true, sentence–sentence J is reweighted by current word focus (v_W) |
| `propagation_converge_tol` | `1e-4` | If set, run layers until L1 change < tol instead of fixed hops (default: converge) |
| `propagation_overlay_path` | `null` | Path to propagation_overlay.json (learnable edge boosts from feedback) |
| `output_dict_boost` | `0.25` | Boost for dictionary terms in output head (one-shot and autoregressive) |
| `use_definition_words_in_activation` | `true` | Seed v_W with token IDs from definition text (definition-aware propagation) |
| `definition_word_weight` | `0.5` | Weight for definition-word tokens added to initial activation |

## Config

- **config/paths.json** – `dictionary_path`, `webster_json_path` (optional; Webster JSON overrides Basis when set), `scratchllm_path`, `align_data_dir`
- **config/anchor.json** – `use_dictionary`, `use_graph_llm`, `use_corpus_graph`, `use_scratchllm`, `include_definitions_in_response`, `system_prompt`, `conversation_turn_limit`, `streaming_max_chunk_chars`, `genre_ids`, `use_grammar`, `grammar_rules_path`, `grammar_command`, `grammar_examples_path`, `use_naturalize`, `naturalize_max_tokens`, `naturalize_context_length`, `use_query_token_ids`, `use_attention_loop`, `attention_loop_hops` (default 4), `attention_loop_top_k`, `attention_loop_use_weights`, `attention_loop_path_groups`, `attention_loop_next_span`, `attention_loop_max_iter`, `attention_loop_output_format` (default paragraph), `attention_loop_paragraph_max_chars`, `use_graph_vectors` (default true), `graph_vectors_boost`, `feedback_path`, `feedback_weights_path`, `feedback_boost`, `use_style_sentences`, `use_critic`, `critic_accept_threshold`, `critic_low_warn_threshold`, `default_genre_id`, `register`, `next_sentence_mode`, `corpus_next_sentences_top_k`, `corpus_hybrid_context_length`, `corpus_hybrid_beta`, `corpus_max_tokens`

Env overrides: `ANCHOR_DICTIONARY_PATH`, `ANCHOR_DATA_DIR`.

## Documentation

- **Math model:** See [docs/UNIFIED_MATH_MODEL.md](docs/UNIFIED_MATH_MODEL.md) for a unified mathematical formulation of the pipeline and a comparison with LLMs.
- **Data math:** See [docs/ANCHOR_LLM_DATA_MATH.md](docs/ANCHOR_LLM_DATA_MATH.md) for a mathematical definition of Anchor data vs LLM data and a compare-and-contrast.

## Project layout

- **anchor/engine.py** – AnchorEngine: concept -> style -> generate -> critic
- **anchor/critic.py** – Dictionary score and accept/warn/reject
- **anchor/retrieval.py** – Concept bundle, style sentences, and graph-based retrieval (Option C); supports multi-genre via `genre_ids`
- **anchor/grammar.py** – Optional grammar rewrite after generation (rules file or external command)
- **anchor/naturalize.py** – Optional post-step: extend response tail using graph next-token patterns (use_naturalize)
- **anchor/generator.py** – Stub, optional scratchLLM/Align, or corpus (next-sentence or hybrid next-token); ScratchLLM can include grammar examples from `grammar_examples_path`
- **anchor/corpus_vocab.py** – Vocabulary build and sentence encoding (Option C)
- **anchor/corpus_graph.py** – Word/sentence graph build and load; by-product transition matrix (Option C)
- **anchor/next_sentence.py** – Next-sentence retrieval from graph (Option C)
- **anchor/next_token.py** – Retrieval + bigram hybrid next-token distribution and sampling
- **anchor/graph_attention.py** – Graph attention loop: activate from query, traverse with optional edge weights and next-word propagation, detect pattern (primary + secondary path groups), optional next-span from similar sentences, refine answer (grounded)
- **anchor/corpus_model.py** – Load and sample from by-product corpus model (1-layer LM)
- **anchor/webster_engine.py** – Webster JSON adapter: load word→definition JSON, implement `get_context_for_description` for concept bundle and critic
- **anchor/wire.py** – Load config and dictionary (Basis or Webster; single place that touches external repos)
- **anchor/corpus_cache.py** – Cache for graph, vocab, and encoded index (speeds up repeated queries in the same session)
- **run_anchor.py** – CLI entry point
- **run_anchor_gui.py** – Tkinter GUI (question + response; Ctrl+Enter to submit)
- **scripts/build_corpus.py** – Build combined corpus with genre tags (from local files)
- **scripts/build_corpus_from_hf.py** – Build corpus from Hugging Face datasets (OpenSubtitles, C4, etc.; requires `datasets`)
- **scripts/build_corpus_from_webster.py** – Ingest Webster dictionary JSON into sentences.jsonl (genre_id e.g. `definitional`); use `--append` to merge with existing corpus
- **scripts/build_corpus_from_dictionary.py** – Ingest dictionary repo compiled corpora (ConceptNet, GooAQ 3M+, Stack, Stdlib) into sentences.jsonl; use after `compile_corpus.py` in the dictionary repo
- **scripts/build_vocab.py** – Build vocab and encoded_sentences from corpus
- **scripts/encode_dictionary.py** – Encode dictionary definitions as token ID sequences (Webster or definitions file); output for build_graph merge
- **scripts/build_graph.py** – Build corpus graph (with inverted index) and optional corpus_model.json; merges encoded_dictionary.jsonl if present
- **scripts/build_vectors.py** – Build word vectors from graph for optional vector geometry (requires graph.json; optional scipy for SVD)
- **anchor/graph_vectors.py** – Load word vectors and boost sentence visits by query–sentence similarity when use_graph_vectors is true
- **anchor/feedback.py** – Record accept/reject feedback; load feedback_weights and boost sentence visits for adaptation
- **scripts/build_feedback_weights.py** – Build feedback_weights.json from feedback.jsonl (accepted responses) for Graph LLM adaptation
- **scripts/build_propagation_overlay.py** – Build propagation_overlay.json from feedback_weights for learnable propagation edges
