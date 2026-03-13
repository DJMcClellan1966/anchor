# Anchor: Formal Theorems and Proofs

This document gives formal definitions, theorem statements, and proofs (or proof sketches) for five results implied by the Anchor/LLM data math. It follows the order of attack: Traceability (4), Convergence (1), Information (3), Equivalence (2), Minimal structure (5). Definitions here align with [ANCHOR_LLM_DATA_MATH.md](ANCHOR_LLM_DATA_MATH.md) and [UNIFIED_MATH_MODEL.md](UNIFIED_MATH_MODEL.md).

---

## Notation and primitives

- **V** = {0, …, *V*−1} = word (token) ID set; **S** = sentence ID set.
- **W** = set of word types (strings); **τ : W → V**, **τ**⁻¹ : **V** → **W**.
- **D** ⊆ **W** × strings: **D**(*t*) = definition text for term *t* when *t* ∈ dom(**D**).
- **t**_s ∈ **V*** = token sequence for sentence *s*; **W**(*s*) = set of word IDs in sentence *s*.
- **P**(*w*′ | *w*) = row-stochastic transition on **V**; **J**(*s*, *s*′) = sentence–sentence similarity (e.g. Jaccard).
- **Encoded index:** enc(*s*) = {text(*s*), genre_id(*s*), source(*s*), term(*s*)}; text(*s*) is the surface string for sentence *s*.
- **C**(*q*) = concept bundle for query *q*: terms ⊆ **W**, definitions = **D** restricted to those terms; **T**(*q*) = {**τ**(*t*) : *t* ∈ terms(**C**(*q*))} ⊆ **V**.

---

## Stage 1: Formal pipeline and Theorem 4 (Traceability)

### 1.1 Formal pipeline

**Activate**(*q*, **C**(*q*), **x**_q):
- **A**_W^(0) = **T**(*q*) ∪ {*x*_q[*i*] : *i*} (terms and query token IDs).
- **A**_S^(0) = ⋃_{*w* ∈ **A**_W^(0)} **S**(*w*); if **A**_S^(0) = ∅ then **A**_S^(0) = **S**.
- Output: (**A**_W^(0), **A**_S^(0)).

**Propagate**(**v**_W, **v**_S; **P**, **J**, *H*):
- One hop: word→sentence, sentence→sentence (J), sentence→word, word→word (P); repeat *H* times (or until convergence if defined). Optional normalization. Output: (**v**_W, **v**_S).

**Pattern**(**v**_W, **v**_S; *K*):
- **W**_top = top *K* words by **v**_W; **S**_top = top *K* sentences by **v**_S.
- Optional: **W**_sec, **S**_sec = next *K* by score not in top; next_span = additional sentence IDs from similar sentences.
- Output: (**W**_top, **S**_top, **W**_sec, **S**_sec, next_span) (or a subset if secondary/next_span omitted).

**Refine**(**W**_top, **S**_top, **W**_sec, **S**_sec, next_span; **C**(*q*), enc, *max_def*, *max_sent*):  
(This corresponds to `refine_answer` in `anchor/graph_attention.py`.)
- **S**_used = **S**_top ∪ **S**_sec ∪ next_span (with duplicates removed; order may be by **v**_S).
- terms_used = {**τ**⁻¹(*w*) : *w* ∈ **W**_top ∪ **W**_sec} ∩ terms(**C**(*q*)).
- Definition parts: for each *t* ∈ terms_used (up to *max_def*), append **D**(*t*) or enc(*s*) for *s* with term(*s*)=*t* (unified store); each part is a string from **D** or from {text(*s*) : *s* ∈ **S**, term(*s*) defined}.
- Sentence parts: for each *s* ∈ **S**_used** with enc(*s*).genre_id in allowed set (up to *max_sent*), append text(*s*); each part is text(*s*).
- Output *y* = concatenation (with separators) of these parts. No other source of characters: every character in *y* appears in at least one of the appended strings.

### 1.2 Definition of source

- A **token** (or character) in the response string *y* has **source** *s* ∈ **S** if it appears in the substring that was appended from text(*s*) in the refinement step.
- It has **source** *t* ∈ dom(**D**) if it appears in the substring that was appended from **D**(*t*) or from text(*s*) for a sentence *s* with term(*s*) = *t* (definition row in the store).

Refinement is defined so that it **only** concatenates substrings that are either (i) text(*s*) for some *s* ∈ **S**_used**, or (ii) **D**(*t*) (or the definition text from the store for term *t*) for some *t* ∈ terms_used. Hence every character in *y* has at least one source in **S** ∪ dom(**D**).

### 1.3 Theorem 4 (Traceability by construction)

**Theorem 4.** Suppose *y* = Refine(**W**_top, **S**_top, **W**_sec, **S**_sec, next_span; **C**(*q*), enc, *max_def*, *max_sent*) with Refine defined as in §1.1. Then every token (character) in *y* has a source in **S** ∪ dom(**D**). Moreover, the number of distinct sentence sources is at most |**S**_top| + |**S**_sec| + |next_span| ≤ *K* + *K* + |next_span|, and the number of distinct definition sources (terms) is at most *max_def* (and at most |**W**_top ∪ **W**_sec| in terms of pattern words used).

**Proof.** By definition of Refine, *y* is built only by appending, in order:
1. For each of at most *max_def* terms from terms_used: one string that is either **D**(*t*) or text(*s*) for some *s* with term(*s*)=*t*. So each character in that substring has source in **S** (if from store) or dom(**D**) (if from **D**).
2. For each of at most *max_sent* sentences from **S**_used**: the string text(*s*). So each character in that substring has source *s* ∈ **S**.

No other characters are appended. So every character in *y* appears in at least one of these strings and therefore has at least one source in **S** ∪ dom(**D**). The bounds on the number of distinct sources follow from the caps *max_def* and *max_sent* and from the fact that **S**_used** is contained in **S**_top ∪ **S**_sec ∪ next_span. ∎

---

## Stage 2: Theorem 1 (Convergence of propagation)

### 2.1 Propagation as a single map

Let **H** = (**v**_W, **v**_S) with **v**_W : **V** → ℝ≥0, **v**_S : **S** → ℝ≥0. One hop of propagation is the composition of:
- Word → sentence: **v**_S(*s*) += ∑_{*w* ∈ **W**(*s*)} **v**_W(*w*) / |**W**(*s*)|.
- Sentence → sentence: **v**_S(*s*′) += ∑_*s* **v**_S(*s*) **J**(*s*, *s*′) (over stored similar pairs).
- Sentence → word: **v**_W(*w*) += ∑_{*s* : *w* ∈ **W**(*s*)} **v**_S(*s*) / |**W**(*s*)|.
- Word → word: **v**_W(*w*′) += ∑_*w* **v**_W(*w*) **P**(*w*′ | *w*).

Optionally apply normalization so that **v**_W and **v**_S are L1-normalized (sum to 1). Denote this one-step map by **F**(**H**) = **H**'.

**State space:** Take the product of two simplices: **X** = Δ(**V**) × Δ(**S**), i.e. nonnegative **v**_W with ∑_*w* **v**_W(*w*) = 1 and nonnegative **v**_S with ∑_*s* **v**_S(*s*) = 1. Assume **F** is defined so that after normalization **F**(**X**) ⊆ **X** (e.g. by L1-normalizing after the additive updates).

### 2.2 Assumptions

- **P** is row-stochastic: ∑_{*w*′} **P**(*w*′ | *w*) = 1 for all *w*.
- **J** is nonnegative and row-normalized or bounded: for each *s*, ∑_{*s*′} **J**(*s*, *s*′) ≤ 1 (or max row sum ≤ 1 − *δ* for some *δ* > 0).
- Word–sentence and sentence–word steps use normalization by |**W**(*s*)| so that mass is preserved or non-expanding in L1.
- **F** includes a normalization step so that **F**(**H**) ∈ **X** and, under the above, **F** is non-expanding in L1; and there is damping or a factor that makes **F** a strict contraction (e.g. **F**(**H**) = (1−*α*)**H** + *α* · (unnormalized propagation) with *α* ∈ (0,1), then normalize).

### 2.3 Theorem 1 (Convergence)

**Theorem 1.** Under the assumptions in §2.2, the propagation map **F** : **X** → **X** is a contraction in the L1 norm (or has spectral radius < 1 on the relevant subspace). Hence **F** has a unique fixed point **H*** ∈ **X**, and for any **H**^(0) ∈ **X**, the iterates **H**^(ℓ+1) = **F**(**H**^(ℓ)) converge to **H*** as ℓ → ∞, with geometric rate ‖**H**^(ℓ) − **H***‖₁ ≤ *c* · *r*^ℓ for some *c* > 0 and *r* ∈ (0,1).

**Proof (sketch).** **X** is complete (closed subset of ℝ^n). Under row-stochastic **P** and bounded **J**, each sub-step of **F** (word→sentence, sentence→sentence, sentence→word, word→word) is non-expanding in L1. With normalization, **F** maps **X** into **X**. If we add damping (e.g. **H**' = (1−*α*)**H** + *α* · (raw propagation), then normalize), then **F** becomes a strict contraction: ‖**F**(**H**_1) − **F**(**H**_2)‖₁ ≤ (1−*α*)‖**H**_1 − **H**_2‖₁ for two distinct points in **X**, so the contraction factor *r* = 1−*α* < 1. By the Banach fixed-point theorem, **F** has a unique fixed point **H*** and **H**^(ℓ) → **H*** with ‖**H**^(ℓ) − **H***‖₁ ≤ *r*^ℓ ‖**H**^(0) − **H***‖₁. ∎

---

## Stage 3: Theorem 3 (Information and condensation)

### 3.1 Behavior

A **behavior** is a mapping *B* : **V*** × *L* → Δ(**V**) from context (sequences of length ≤ *L*) to a distribution over the next token. Here **V*** × *L* denotes the set of sequences over **V** of length at most *L*.

### 3.2 Lower bound

**Theorem 3a (Lower bound).** Let *B* be a behavior induced by a row-stochastic **P** (e.g. *B*(*x*) = **P**(· | *x*_{−1}) or a propagation+output over **P**, **J**). Let *H* be the entropy rate of the stationary distribution of **P** (or the average entropy of *B*(*x*) over contexts *x*). To represent *B* to within ε in total variation (or KL) over a set of *N* contexts, any representation (explicit or parametric) requires at least Ω(*H* · *N*) bits (or equivalent parameter count) in the limit as ε → 0.

**Proof (sketch).** This follows from rate–distortion or source-coding: approximating a distribution with entropy *H* to within ε in KL or TV requires at least on the order of *H* bits (see standard information-theoretic bounds). Summing over contexts gives the lower bound. ∎

### 3.3 Upper bound

**Theorem 3b (Upper bound).** (a) An explicit representation (**S**, **t**_s, **P**, **J**) of size *O*(|**V**|² + |**S**|²) can realize any behavior that is the limit of propagation + output head over that graph. (b) A transformer (or MLP) of size poly(|**V**|, *L*, 1/ε) can ε-approximate any such behavior in total variation (or KL).

**Proof (sketch).** (a) The graph **P**, **J** and the sequence store have the stated size; propagation + output head compute the next-token distribution in a finite number of steps. (b) Universal approximation: a sufficiently large transformer can approximate any continuous function from a compact set (e.g. context embeddings) to the simplex Δ(**V**) to within ε. ∎

---

## Stage 4: Theorem 2 (Equivalence / approximation)

### 4.1 Realizes

A system **realizes** a behavior *B* (or ε-realizes it) if for every context *x*, the system’s next-token distribution equals *B*(*x*) (or is within ε in TV or KL).

### 4.2 Theorem 2a (Anchor → LLM)

**Theorem 2a.** For every Anchor instance ( **V**, **S**, **t**_s, **P**, **J**, **D**, enc ) of size *N* (bounded |**V**|, |**S**|, and description length), the one-shot pipeline (activate → propagate → pattern → refine) induces a finite function from (query, concept_bundle) to a response string, and the next-token (or next-sentence) distribution induced by the pattern step can be viewed as a distribution over **V**. There exists a transformer (or MLP) of size poly(*N*, 1/ε) whose next-token distribution ε-approximates this Anchor-induced distribution in total variation.

**Proof (sketch).** The Anchor pipeline is a finite computation on finite inputs; its output is determined by (**v**_W, **v**_S) after propagation and the pattern step, which yields a distribution over **V** (e.g. from **v**_W after normalization). That is a continuous (in fact piecewise-linear) function from the compact set of inputs to Δ(**V**). By universal approximation, a transformer of size poly(*N*, 1/ε) can approximate this function to within ε in TV. ∎

### 4.3 Theorem 2b (LLM → Anchor)

**Theorem 2b.** For every transformer **T** with context length *L* and vocabulary **V**, there exists an explicit structure (**S**, **t**_s, **P**, **J**) such that propagation + output head ε-approximates **T**’s next-token distribution. The size of (**S**, **P**, **J**) can be chosen at most *O*(|**V**|^(*L*)) (e.g. one sentence or state per context, transitions encoded in **P**, **J**).

**Proof (sketch).** The transformer **T** computes a function from context *x* ∈ **V***_≤*L* to a distribution over **V**. The set of contexts is finite (size at most |**V**|^(*L*)). For each context, define a “sentence” or state whose text (or identity) encodes that context; let **P** and **J** encode the transition and similarity so that propagation + output head reproduce (or ε-approximate) **T**’s next-token distribution. Construction yields an explicit structure of size *O*(|**V**|^(*L*)). ∎

---

## Stage 5: Theorem 5 (Minimal structure)

### 5.1 Minimal

A structure (**S**, **t**_s, **P**, **J**, **D**) **realizes** a behavior *B* (or a finite set of query–answer pairs) if the Anchor pipeline with that structure produces responses that match *B* (or the given pairs). **Minimal** means minimizing |**S**| (or total description length of the structure) over all structures that realize the behavior (or ε-realize it).

### 5.2 Theorem 5 (Existence and bounds)

**Theorem 5.** (a) For any finite behavior (e.g. finite set of query–answer pairs or next-token distributions on a finite set of contexts), the set of explicit structures that realize it (or ε-realize it) is non-empty. Hence a minimal such structure exists (minimum |**S**| or minimum description length).  
(b) For the class of behaviors given by *m* distinct query–answer pairs, min |**S**| ≥ *m* (at least one sentence per distinct answer), and min |**S**| ≤ *f*(*m*, |**V**|) for some function *f* (e.g. one sentence per pair plus auxiliary sentences for **P**, **J**).

**Proof (sketch).** (a) Take one sentence per query–answer pair (the answer as text(*s*)), and define **P**, **J** so that activation and propagation select those sentences. So a realizing structure exists. The set of structures with |**S**| ≤ *M* is finite for each *M*; so the set of structures that realize the given finite behavior is non-empty and has a minimum (by taking *M* large enough and minimizing over that set).  
(b) Lower bound: if there are *m* distinct answers, we need at least *m* distinct text strings, hence at least *m* sentence IDs. Upper bound: construct one sentence per pair and add a bounded number of auxiliary nodes/edges so that **P**, **J** route correctly; *f* is polynomial in *m* and |**V**|. ∎

---

## Summary

| Theorem | Statement |
|--------|-----------|
| **4** | Every token in the Anchor response has a source in **S** ∪ dom(**D**); number of sources bounded by *K*, *max_def*, *max_sent*, next_span. |
| **1** | Under **P** row-stochastic and **J** bounded/damped, propagation **F** has a unique fixed point and iterates converge geometrically. |
| **3** | Lower: representation size Ω(*H*); Upper: explicit *O*(|**V**|² + |**S**|²), parametric poly(|**V**|, *L*, 1/ε). |
| **2a** | Anchor-induced behavior is ε-approximable by a transformer of size poly(*N*, 1/ε). |
| **2b** | Transformer is ε-approximable by an explicit structure of size *O*(|**V**|^(*L*)). |
| **5** | Minimal realizing structure exists; for *m* query–answer pairs, min |**S**| between *m* and *f*(*m*, |**V**|). |
