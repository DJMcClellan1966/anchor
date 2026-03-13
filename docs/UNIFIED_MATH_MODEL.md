# Unified Mathematical Model: Anchor and LLM Comparison

This document formalizes the Anchor pipeline as a single mathematical model (vocabulary, graph, activation, propagation, pattern, refinement, critic), describes latent-structure and analytical "powers," formulates a standard LLM in the same style, and compares the two.

---

## 1. Anchor: Mathematical Objects and Pipeline

### Vocabulary and tokenization

- **W** = set of word types (strings).
- **V** = {0, …, V−1} = set of word IDs (vocabulary size V).
- **τ : W → V** = word-to-id map (e.g. `word_to_id`). **τ⁻¹** = id-to-word.
- **φ** = tokenize: string → list of tokens (e.g. words and punctuation).
- Query *q* → **x**_q = [τ(*w*) : *w* ∈ φ(*q*), τ(*w*) defined] = query token ID sequence.

### Dictionary / concept bundle

- **D** = dictionary (term → definition).
- Concept map **C**(*q*) = terms + definitions returned for query *q* (from engine).
- **T**(*q*) ⊆ **V** = set of word IDs for terms in **C**(*q*) (via τ).

### Graph

- **S** = set of sentence IDs.
- For each *s* ∈ **S**, **t**_s ∈ **V*** = token sequence for sentence *s* (sentence_words).
- **W**(*s*) = {**t**_s[*i*]} = set of word IDs in sentence *s*.
- **S**(*w*) = { *s* : *w* ∈ **W**(*s*) } = sentences containing word *w*.
- **N**_w(*w*′ | *w*) = next-word count (word_next): count of *w* → *w*′ in corpus.
- **P**(*w*′ | *w*) = normalized transition (row-stochastic); e.g. **P**(*w*′|*w*) = (**N**_w(*w*′,*w*) + *α*) / (∑_*w*″ **N**_w(*w*″,*w*) + *V* *α*) with smoothing *α*.
- Co(*w*) ⊆ **V** = co-occurring word IDs (word_cooccurrence).
- **J**(*s*, *s*′) = Jaccard similarity of **W**(*s*) and **W**(*s*′) (sentence_similar); stored as top-*k* similar pairs per sentence.

### Activation

- **A**_W^(0) = **T**(*q*) ∪ { **x**_q[*i*] } (terms union query token IDs).
- **A**_S^(0) = ⋃_{*w* ∈ **A**_W^(0)} **S**(*w*). If **A**_S^(0) = ∅, set **A**_S^(0) = **S** (fallback: all sentences).

### Propagation (traverse_loops)

- State: **v**_W : **V** → ℝ≥0, **v**_S : **S** → ℝ≥0.
- Init: **v**_W(*w*) = 1 for *w* ∈ **A**_W^(0), **v**_S(*s*) = 1 for *s* ∈ **A**_S^(0); else 0.
- For *h* = 1, …, *H*−1 (num_hops):
  - **Word → sentence:** **v**_S(*s*) += ∑_{*w* ∈ **W**(*s*)} **v**_W(*w*) / |**W**(*s*)| (or 1 if no length norm).
  - **Sentence → sentence:** **v**_S(*s*′) += ∑_*s* **v**_S(*s*) · **J**(*s*, *s*′) (over top-*k* similar *s*′ per *s*).
  - **Sentence → word:** **v**_W(*w*) += ∑_{*s* : *w* ∈ **W**(*s*)} **v**_S(*s*) / |**W**(*s*)|.
  - **Word → word (next-token):** **v**_W(*w*′) += ∑_*w* **v**_W(*w*) **P**(*w*′ | *w*).
- Optional: genre filter as mask on **S**; same recurrence restricted to allowed *s*.
- **Optional extensions (refactored implementation):** (1) **Co(*w*)** word→word: spread **v**_W to co-occurring words. (2) **P**_prev backward: **v**_W(*w*) += ∑_{*w*′} **v**_W(*w*′) **P**_prev(*w*|*w*′) (predecessor counts from corpus). (3) **Content-dependent J:** reweight **J**(*s*,*s*′) by overlap of **v**_W with **W**(*s*′) so sentence–sentence flow depends on current word focus. (4) **Run until convergence:** iterate until ‖**H**^(ℓ) − **H**^(ℓ−1)‖ < tol instead of fixed *H*. (5) **Output head reweight:** boost dictionary terms in *p* before softmax. (6) **Learnable overlay:** additive edge boosts (word–word, sentence–sentence) loaded from file and updated from feedback.

### Pattern (detect_pattern)

- **W**_top = top *K* words by **v**_W; **S**_top = top *K* sentences by **v**_S (with optional min_visits threshold).
- Optional secondary group: next *K* by score not in top (**W**_sec, **S**_sec).

### Refinement (refine_answer)

- Map **W**_top ∪ **S**_top (and optional **W**_sec, **S**_sec, next_span) to text: definitions from **C**(*q*) for terms in **W**_top; sentence texts from encoded index for **S**_top; order by **v**_S and optional secondary/next_span. Output = single string *y* (list or paragraph format).

### Critic

- Content terms terms(*y*) ⊆ **W** (extract_content_terms from *y*).
- Grounded *G* = { *w* ∈ terms(*y*) : *w* in dictionary **D** }.
- Score *σ* = |*G*| / max(|terms(*y*)|, 1). Decision = accept / warn / reject by thresholds.

### Unified pipeline (one formula)

- *y* = Refine( Pattern( Propagate( Activate(*q*, **C**(*q*), **x**_q); *G*, **P**, **J**, *H* ) ); **C**(*q*), enc ).
- Critic: *σ*, decision = Critic(*y*; **D**).

### Anchor in LLM-like form

The same pipeline can be written in the same shape as an LLM: **Embed** → **Layers** → **Output head**.

- **Embed:** **H**^(0) = (**v**_W^(0), **v**_S^(0)) = Embed_anchor(*q*, **C**(*q*), **x**_q): initial activation (terms ∪ query token IDs → weight 1 on word and sentence nodes).
- **Layers:** For ℓ = 1, …, *L*: (**v**_W, **v**_S) = propagation_layer(**v**_W, **v**_S; *G*, **P**, **J**); optionally **v**_W = Norm(**v**_W), **v**_S = Norm(**v**_S) so each step keeps a distribution (attention-like).
- **Output head:** **h** = **v**_W^(L) (word part of state); *p* = softmax(**h**) = L1-normalize(**v**_W) as distribution over words.

**One-shot (current):** Use *p* to take top *K* words (and top *K* sentences by **v**_S), then Refine(definitions + sentences) → *y*.

**Autoregressive:** Loop: context = query token IDs; repeat: Embed(context) → Layers → *p* = output_head(**v**_W) → sample next token from *p* → append to context; until stop (max_tokens or sentence-ending token). *y* = concatenation of generated tokens (optional definitions prepended).

---

## 2. Latent Ideas and Powers (Anchor)

### Transition matrix P

**P**(*w*′ | *w*) defines a Markov chain on **V**. Powers **P**^ℓ give ℓ-step transition probabilities. If the chain is ergodic, the stationary distribution *π* (e.g. *π* = *π* **P**) indicates "central" or "hub" words. Leading eigenvectors of **P** (or of a normalized Laplacian derived from it) can reveal recurrent or thematic structure. **Latent idea:** word importance and recurrence structure.

### Graph structure

The word–sentence incidence and sentence–sentence similarity **J** form a bipartite and a weighted graph. Spectral analysis (e.g. Laplacian eigenvectors) or clustering on the sentence–sentence graph can reveal themes or sentence communities. **Latent idea:** document/sentence latent structure.

### Visit distributions v_W, v_S

After propagation, **v**_W and **v**_S are the result of a fixed number of recurrence steps. They can be interpreted as a query-induced "attention" or importance distribution over words and sentences. **Latent idea:** query-specific salience.

### Information / entropy

The entropy *H*(**P**(·|*w*)) = − ∑_*w*′ **P**(*w*′|*w*) log **P**(*w*′|*w*) measures predictability from *w*. Entropy of **v**_W (normalized to a distribution) measures concentration of the visit mass. **Latent idea:** predictability and focus of the model.

---

## 3. LLM in the Same Mathematical Style

Standard autoregressive decoder-only transformer:

### Vocabulary

- **V** = token IDs; *V* = vocab size; *τ*, *τ*⁻¹; max sequence length *L*.

### Embedding

- **E** ∈ ℝ^{V×d}; position encoding **P** (e.g. sinusoidal or learned).
- Input sequence **x** ∈ **V**^L → **X**^(0) = **E**[**x**] + **P** (token embeddings plus positions).

### Layers ℓ = 1, …, L_layer

- **Q**^(ℓ), **K**^(ℓ), **V**^(ℓ) = **X**^(ℓ−1) **W**_Q, **X**^(ℓ−1) **W**_K, **X**^(ℓ−1) **W**_V.
- **A**^(ℓ) = softmax(**Q** **K**^T / √*d*) **V** (scaled dot-product attention); then residual + layer norm.
- **X**^(ℓ) = Norm(**X**^(ℓ−1) + MLP(**X**^(ℓ−1))).

### Output head

- **h** = **X**^(L_layer)[−1] (last position).
- Logits **z** = **h** **E**^T (or separate output projection); *p*(next token) = softmax(**z**).

### Forward pass

- *p*(*y*_t | *y*_{<t}); full response = autoregressive sampling or argmax from *p*.

---

## 4. Compare and Contrast

| Axis | Anchor | LLM |
|------|--------|-----|
| **Representation** | Discrete sets + counts + graph (words, sentences, **P**); IDs in **V**, **S**. | Continuous embeddings and hidden states in ℝ^d; token IDs only at boundaries. |
| **Computation** | Fixed recurrence: activation → propagation → pattern → refinement. No learned weights in the graph path. | Stacked attention + MLP layers (learned **W**_Q, **W**_K, **W**_V, MLP); then softmax for next-token. |
| **Knowledge** | Explicit dictionary **D** + graph *G* + encoded corpus; retrieval-based. | Parametric: all knowledge in weights; no explicit retrieval in the core forward pass. |
| **Query use** | Query drives activation and propagation to produce **v**_W, **v**_S; pattern selects top nodes; refinement maps to text. | Query (and context) as prefix **x**; then autoregressive generation from *p*(· | **x**, *y*_{<t}). |
| **Latent structure** | **P**, **J**, and visit distributions; interpretable discrete structure (words, sentences, counts). | Hidden dimensions and attention weights; continuous; often interpreted post hoc (probes, attention visualization). |
| **Groundedness** | Critic checks response terms against **D**; response is composed from definitions and corpus sentences only. | No built-in symbolic grounding; can add RAG or tools that retrieve into context. |

**Summary:** Anchor is a discrete, graph-based, retrieval-and-compose model with an explicit dictionary and visit propagation; the LLM is a continuous, sequence-based, parametric autoregressive model. Both can be written as a single end-to-end formula (Anchor: Activate → Propagate → Pattern → Refine; LLM: Embed → Layers → Softmax), but the former uses sets and counts on a fixed graph, the latter uses matrix multiplies and softmax over a learned representation.
