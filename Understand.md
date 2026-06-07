# Understand.md — Learning the Transformer Paper Step by Step

> **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
> **Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google Brain, 2017)  
> **This guide:** Explains the paper, why it matters, how we use Transformers today, and how every file in this project maps to the research.

---

## Table of Contents

1. [The Big Picture — What Problem Did the Paper Solve?](#1-the-big-picture--what-problem-did-the-paper-solve)
2. [Why We Use Transformers](#2-why-we-use-transformers)
3. [Paper Architecture — Section by Section](#3-paper-architecture--section-by-section)
4. [How We Implement It — File by File](#4-how-we-implement-it--file-by-file)
5. [Step-by-Step Learning Path (Recommended Order)](#5-step-by-step-learning-path-recommended-order)
6. [Design Choices: Could We Use Something Else?](#6-design-choices-could-we-use-something-else)
7. [Where Transformers Are Used Today](#7-where-transformers-are-used-today)
8. [Use Cases and Key Features](#8-use-cases-and-key-features)
9. [Common Interview Questions](#9-common-interview-questions)
10. [What This Project Does vs. What the Paper Did](#10-what-this-project-does-vs-what-the-paper-did)

---

## 1. The Big Picture — What Problem Did the Paper Solve?

Before 2017, the best models for translation and language tasks were mostly **RNNs** (Recurrent Neural Networks) and **LSTMs**:

```
Token 1 → Token 2 → Token 3 → Token 4 → ...
   ↓         ↓         ↓         ↓
 hidden    hidden    hidden    hidden
```

Each token had to wait for the previous one. That caused two big problems:

| Problem | What it means |
|---------|----------------|
| **Slow training** | Tokens are processed one-by-one — hard to parallelize on GPU |
| **Long-range memory** | Information from early tokens gets diluted over many steps (vanishing gradients) |

The 2017 paper asked: **Can we remove recurrence entirely and still model language well?**

Their answer: **Yes — use Attention.**

```
Every token can look at every other token directly:

  hello ──────────────┐
     ↓                ↓
  world ──── attends to ──── hello
     ↓                ↓
  ! ──────────────────┘
```

That idea became the **Transformer**. It is the foundation of GPT, BERT, Claude, Gemini, and most modern AI systems.

---

## 2. Why We Use Transformers

### 2.1 Core advantages

| Advantage | Explanation |
|-----------|-------------|
| **Parallelism** | All tokens in a sequence can be processed at once during training |
| **Long-range dependencies** | Any token can directly attend to any other token in O(1) hops |
| **Interpretability** | Attention weights show *which* tokens the model focuses on |
| **Scalability** | Stack more layers, more heads, more data → better results (proven at scale) |
| **One architecture, many tasks** | Translation, summarization, Q&A, code generation, vision, audio |

### 2.2 The trade-off

Attention is **O(L²)** in sequence length L — every token compares to every other token.

| Sequence length | Attention cost |
|-----------------|----------------|
| L = 128 | Fine |
| L = 4,096 | Heavy but manageable |
| L = 1,000,000 | Impractical without tricks (Flash Attention, sparse attention, sliding windows) |

Modern systems use optimizations (Flash Attention, KV-cache, sparse patterns) to handle this. Our educational project uses short sequences (max 20 tokens), so this is not a concern here.

---

## 3. Paper Architecture — Section by Section

The paper describes an **Encoder-Decoder** model for machine translation (e.g., English → French).

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRANSFORMER                              │
│                                                                  │
│  Source: "hello world"          Target: "bonjour monde"         │
│         │                              │                         │
│         ▼                              ▼                         │
│  ┌─────────────┐              ┌─────────────┐                   │
│  │   ENCODER   │── memory ──▶ │   DECODER   │──▶ output probs  │
│  │  (N layers) │              │  (N layers) │                   │
│  └─────────────┘              └─────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Section 3.2 — Scaled Dot-Product Attention

The heart of the model:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

| Symbol | Meaning | Shape (in our code) |
|--------|---------|---------------------|
| Q | Query — "what am I looking for?" | `(B, H, L_q, d_k)` |
| K | Key — "what do I offer?" | `(B, H, L_k, d_k)` |
| V | Value — "what information do I carry?" | `(B, H, L_v, d_v)` |
| d_k | Dimension per head | 32 (when d_model=128, heads=4) |
| H | Number of heads | 4 |

**Why divide by √d_k?**  
Dot products grow with dimension. Without scaling, softmax saturates (all weight on one token) and gradients vanish.

### 3.2 Section 3.2 — Multi-Head Attention

Instead of one attention function, run **H parallel heads** on different subspaces:

```
head_i = Attention(Q·W_q_i, K·W_k_i, V·W_v_i)
MultiHead = Concat(head_1, ..., head_H) · W_o
```

Each head can learn different relationships (syntax, semantics, position, etc.).

### 3.3 Section 3.3 — Position-wise Feed-Forward Network

A small 2-layer MLP applied **independently to each token**:

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

Same weights for every position — adds non-linearity after attention mixes tokens.

### 3.4 Section 3.4 — Embeddings and Softmax

- Token embeddings map word IDs → vectors of size `d_model`
- Paper multiplies embeddings by `√d_model` before adding positional encoding
- Final linear layer projects decoder output → vocabulary size → softmax for probabilities

### 3.5 Section 3.5 — Positional Encoding

Attention alone does not know word order. "dog bites man" and "man bites dog" would look the same.

The paper adds fixed sinusoidal encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Final input to the model: `Embedding(token) + PositionalEncoding(position)`

### 3.6 Residual Connections + Layer Normalization

Each sub-layer (attention, FFN) uses:

```
output = LayerNorm(x + Sublayer(x))
```

This is **Post-LayerNorm** (norm after the residual add), matching the original paper.

| Without residuals | With residuals |
|-------------------|----------------|
| Gradients vanish in deep networks | Gradients flow through skip connections |
| Hard to train 6+ layers | Stable training at depth |

---

## 4. How We Implement It — File by File

Below is a walkthrough of **every important file** in this project and how it connects to the paper.

---

### 4.1 `config.py` — Hyperparameters

**Paper reference:** Section 5.4 (training details)

**What it does:** Central place for all model and training settings.

| Parameter | Our value | Paper value | Why different? |
|-----------|-----------|-------------|----------------|
| `D_MODEL` | 128 | 512 | Smaller = faster on laptop |
| `N_HEADS` | 4 | 8 | Must divide d_model evenly |
| `D_FF` | 512 | 2048 | Typically 4× d_model |
| `N_LAYERS` | 2 | 6 | Fewer layers for toy data |
| `DROPOUT` | 0.0 | 0.1 | Tiny dataset — we want memorization for demo |
| `EPOCHS` | 300 | — | Many passes over 8 sentences |

**When you read code:** Start here to understand model size before diving into modules.

---

### 4.2 `models/attention.py` — Attention (Paper §3.2)

**Paper reference:** Section 3.2, Equation (1)

**What it implements:**

| Function / Class | Purpose |
|------------------|---------|
| `scaled_dot_product_attention()` | Core Q·Kᵀ/√d_k → softmax → ·V |
| `create_padding_mask()` | Hide `<pad>` tokens from attention |
| `create_causal_mask()` | Prevent decoder from seeing future tokens |
| `MultiHeadAttention` | W_q, W_k, W_v, W_o projections + head split/merge |

**Data flow:**

```
Input x: (batch, seq_len, d_model)
    │
    ├─ W_q → split heads → Q: (B, H, L, d_k)
    ├─ W_k → split heads → K: (B, H, L, d_k)
    └─ W_v → split heads → V: (B, H, L, d_v)
              │
              ▼
    scaled_dot_product_attention(Q, K, V, mask)
              │
              ▼
    merge heads → W_o → output: (B, L, d_model)
```

**Notebook:** `notebooks/01_attention.ipynb`, `02_multihead.ipynb`

---

### 4.3 `models/positional_encoding.py` — Positional Encoding (Paper §3.5)

**Paper reference:** Section 3.5, Equations (3) and (4)

**What it implements:**

- Precomputes sin/cos table of shape `(1, max_len, d_model)`
- Adds to embeddings: `x + pe[:, :seq_len, :]`
- Applies dropout after addition

**Helper functions:**

- `plot_positional_encoding()` — heatmap for visualization
- `plot_positional_encoding_waves()` — sin/cos curves per dimension

**Notebook:** `notebooks/03_positional_encoding.ipynb`

---

### 4.4 `models/feed_forward.py` — Feed-Forward Network (Paper §3.3)

**Paper reference:** Section 3.3, Equation (2)

**What it implements:**

```
Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model) → Dropout
```

Applied to every token independently — shape stays `(batch, seq_len, d_model)`.

---

### 4.5 `models/encoder.py` — Encoder Stack (Paper §3.1, left side)

**Paper reference:** Section 3.1, Figure 1 (encoder stack)

**What it implements:**

| Class | Role |
|-------|------|
| `EncoderLayer` | One layer: Self-Attn → Add&Norm → FFN → Add&Norm |
| `Encoder` | Embedding × √d_model → PosEnc → N × EncoderLayer |

**Output:** **Memory** tensor `(batch, src_len, d_model)` — the encoded source sentence that the decoder will attend to.

**Mask used:** Padding mask only (encoder sees all real source tokens).

---

### 4.6 `models/decoder.py` — Decoder Stack (Paper §3.1, right side)

**Paper reference:** Section 3.1, Figure 1 (decoder stack)

**What it implements:**

| Class | Role |
|-------|------|
| `DecoderLayer` | Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm |
| `Decoder` | Embedding → PosEnc → N × DecoderLayer |

**Three types of attention in the decoder:**

| Attention type | Q from | K, V from | Mask |
|----------------|--------|-----------|------|
| Masked self-attention | Decoder | Decoder | Causal (no future) |
| Cross-attention | Decoder | Encoder memory | Source padding |

---

### 4.7 `models/transformer.py` — Full Model (Paper §3, Figure 1)

**Paper reference:** Full architecture, Figure 1

**What it implements:**

| Method | When used |
|--------|-----------|
| `forward(src, tgt)` | Training with teacher forcing |
| `encode(src)` | Run encoder only |
| `decode(tgt, memory, src)` | Run decoder given encoder output |
| `greedy_decode(src)` | Inference — pick highest-probability token each step |
| `beam_search_decode(src)` | Inference — keep top-K partial sequences (bonus) |

**Teacher forcing (training):**

```
Decoder input:  <sos> bonjour <eos> <pad> ...
Predict targets: bonjour <eos>  (shifted by 1)
```

The decoder always sees the **correct** previous tokens during training, which stabilizes learning.

**Autoregressive decoding (inference):**

```
Step 1: <sos>           → predict "bonjour"
Step 2: <sos> bonjour   → predict "<eos>"
Step 3: stop
```

---

### 4.8 `models/gpt_decoder.py` — Bonus: Decoder-Only Model

**Paper reference:** Decoder stack only (no encoder)

**What it is:** A simplified **GPT-style** model — only masked self-attention, no cross-attention.

| Full Transformer (this project) | GPT-style (bonus) |
|-------------------------------|-------------------|
| Encoder + Decoder | Decoder only |
| Cross-attention to source | No encoder |
| Best for translation | Best for text generation |

Modern LLMs (GPT-4, Llama, Claude) use decoder-only architectures at massive scale.

---

### 4.9 `dataset.py` — Data Pipeline

**Paper reference:** Section 5.1 (WMT translation task) — we use a toy subset

**What it implements:**

| Component | Purpose |
|-----------|---------|
| `TRANSLATION_PAIRS` | 8 English→French sentence pairs |
| `Vocabulary` | Word-level token ↔ ID mapping |
| `TranslationDataset` | Returns padded `(src_ids, tgt_ids)` tensors |
| `get_dataloaders()` | Train/val split + DataLoader |

**Special tokens:**

| Token | Purpose |
|-------|---------|
| `<pad>` | Fill shorter sequences to fixed length |
| `<sos>` | Start of sequence (decoder input) |
| `<eos>` | End of sequence (stop decoding) |
| `<unk>` | Unknown word fallback |

---

### 4.10 `train.py` — Training Loop

**Paper reference:** Section 5.3 (optimizer, learning rate schedule)

**What it does step by step:**

```
1. Load data and build vocabularies
2. Create Transformer model
3. For each epoch:
     a. Forward pass: logits = model(src, tgt)
     b. Loss = CrossEntropy(logits, shifted_tgt), ignore <pad>
     c. Backprop + Adam optimizer step
4. Save best checkpoint to checkpoints/best_model.pt
5. Export training_history.json and attention_snapshots.json
```

**Loss function:** Standard cross-entropy over target vocabulary, ignoring padding positions.

---

### 4.11 `inference.py` — Translation at Runtime

**What it does:**

```
1. Load checkpoint + vocabularies
2. Encode source sentence → token IDs
3. greedy_decode() or beam_search_decode()
4. Decode token IDs → French words
```

**CLI examples:**

```bash
python inference.py --src "hello"
python inference.py --src "thank you" --beam 3
```

---

### 4.12 `visualizations/` — Plots and HTML Showcase

| File | Purpose |
|------|---------|
| `attention_heatmaps.py` | Matplotlib heatmaps, loss curves, PE plots |
| `generate_showcase.py` | Builds single-file `demo/showcase.html` |

**Showcase HTML includes:**

- Architecture diagram (SVG)
- Interactive Plotly attention heatmaps
- Positional encoding plot (embedded PNG)
- Training loss curve
- Translation results table

---

### 4.13 `demo/streamlit_app.py` — Interactive Demo

Live translation + attention visualization in the browser.

---

### 4.14 `notebooks/` — Hands-On Tutorials

| Notebook | Topic |
|----------|-------|
| `01_attention.ipynb` | Scaled dot-product attention, masks, heatmaps |
| `02_multihead.ipynb` | Multi-head split/merge, 4-head visualization |
| `03_positional_encoding.ipynb` | Sin/cos waves and heatmap |
| `04_transformer_training.ipynb` | End-to-end training walkthrough |

---

## 5. Step-by-Step Learning Path (Recommended Order)

Follow this order as a **beginner researcher**:

```
Phase 1 — Foundations
├── Read: Sections 1-3 of this file (Big Picture + Paper Architecture)
├── Code: models/attention.py
├── Run:  notebooks/01_attention.ipynb
└── Ask:  "Why scale by √d_k?" — run the numeric demo in the notebook

Phase 2 — Building Blocks
├── Code: models/positional_encoding.py
├── Run:  notebooks/03_positional_encoding.ipynb
├── Code: models/feed_forward.py
└── Ask:  "Why do we need positional encoding if attention already mixes tokens?"

Phase 3 — Encoder & Decoder
├── Code: models/encoder.py → trace shapes on paper
├── Code: models/decoder.py → understand causal vs cross-attention masks
└── Ask:  "What is the difference between self-attention and cross-attention?"

Phase 4 — Full Pipeline
├── Code: models/transformer.py
├── Code: dataset.py + train.py
├── Run:  python train.py
├── Run:  python inference.py --src "hello"
└── Ask:  "What is teacher forcing and why do we need it?"

Phase 5 — Visualization & Intuition
├── Run:  python visualizations/generate_showcase.py
├── Open: demo/showcase.html
├── Run:  streamlit run demo/streamlit_app.py
└── Ask:  "Which source words does the decoder attend to when generating each French word?"
```

---

## 6. Design Choices: Could We Use Something Else?

This section answers **"Why this? Could we use that instead?"** — common questions researchers ask.

### 6.1 Attention mechanism

| Choice in paper | Alternative | When to use alternative |
|-----------------|-------------|-------------------------|
| Scaled dot-product | Additive attention (Bahdanau) | Rare today; dot-product is faster on GPU |
| Multi-head | Single-head | Single-head is simpler but less expressive |
| Softmax attention | Linear attention, Performer, Mamba | Very long sequences (100K+ tokens) |

### 6.2 Positional encoding

| Choice in paper | Alternative | Used in |
|-----------------|-------------|---------|
| Fixed sin/cos | Learned positional embeddings | GPT, BERT (original) |
| Absolute position | Relative position (T5, Shaw et al.) | Long text, better generalization |
| Added to embedding | Rotary (RoPE) — GPT-NeoX, Llama | Modern LLMs |
| — | ALiBi (attention bias) | BLOOM, some long-context models |

**Our project:** Fixed sin/cos (faithful to the 2017 paper).

### 6.3 Normalization

| Choice in paper | Alternative | Notes |
|-----------------|-------------|-------|
| Post-LayerNorm | Pre-LayerNorm | Pre-LN trains more stably in very deep models (GPT-2+) |
| LayerNorm | RMSNorm | Used in Llama — slightly simpler, no mean centering |

**Our project:** Post-LayerNorm (matches original paper).

### 6.4 Activation in FFN

| Choice in paper | Alternative | Used in |
|-----------------|-------------|---------|
| ReLU | GELU | BERT, GPT-2+ |
| ReLU | SwiGLU | Llama, PaLM — gated FFN |

### 6.5 Architecture variant

| Variant | Best for | Examples |
|---------|----------|----------|
| Encoder-Decoder | Translation, summarization | Original Transformer, T5, BART |
| Encoder-only | Understanding, classification | BERT, RoBERTa |
| Decoder-only | Text generation | GPT, Llama, Claude |

**Our project:** Encoder-Decoder (translation demo). Bonus `gpt_decoder.py` shows decoder-only.

### 6.6 What we deliberately did NOT use (and why)

| Skipped | Why |
|---------|-----|
| `torch.nn.Transformer` | Educational goal — learn by building |
| Hugging Face `transformers` | Same reason — no black boxes |
| Learned positional embeddings | Paper uses sin/cos; we follow the paper |
| Label smoothing | Paper uses it; we skip for simplicity on toy data |
| Learning rate warmup schedule | Paper uses it; we use fixed LR for simplicity |

---

## 7. Where Transformers Are Used Today

The 2017 paper was designed for **machine translation**. Today, Transformers power most of AI.

### 7.1 Natural Language Processing

| Application | Model type | Examples |
|-------------|------------|----------|
| Chat / text generation | Decoder-only | GPT-4, Claude, Gemini, Llama |
| Search / understanding | Encoder-only | BERT, sentence embeddings |
| Translation / summarization | Encoder-Decoder | Google Translate, T5, NLLB |
| Code generation | Decoder-only | GitHub Copilot, Codex |

### 7.2 Beyond text

| Domain | How Transformers are used |
|--------|---------------------------|
| **Vision** | ViT (Vision Transformer) — images as patch sequences |
| **Audio** | Whisper — speech as token sequences |
| **Multimodal** | GPT-4V, Gemini — text + images in one model |
| **Protein folding** | AlphaFold2 — attention over amino acid pairs |
| **Robotics** | RT-2, policy learning over sensor tokens |
| **Time series** | Temporal Fusion Transformer |

### 7.3 Industry impact timeline

```
2017  Transformer paper (translation)
2018  BERT (understanding), GPT-1 (generation)
2020  GPT-3 (175B parameters — scale matters)
2022  ChatGPT (RLHF + instruction tuning)
2023+ GPT-4, Claude, Gemini, open-source Llama/Mistral
```

---

## 8. Use Cases and Key Features

### 8.1 When Transformers shine

- **Sequence-to-sequence tasks** — translation, summarization, dialogue
- **Long-context reasoning** — document Q&A, code analysis
- **Transfer learning** — pre-train on huge corpus, fine-tune on small task
- **Multimodal fusion** — text + image + audio in one model

### 8.2 When Transformers struggle (or need tricks)

| Challenge | Solution used in production |
|-----------|----------------------------|
| Quadratic memory in L | Flash Attention, sparse attention, sliding windows |
| No built-in time/order for streaming | Causal masking, KV-cache |
| Expensive inference | Quantization (INT8/INT4), distillation, speculative decoding |
| Need exact retrieval | RAG (Retrieval-Augmented Generation) on top of LLM |

### 8.3 Key features of the architecture

| Feature | Benefit |
|---------|---------|
| Self-attention | Global context in one layer |
| Multi-head | Parallel representation subspaces |
| Residual connections | Train deep networks |
| Layer normalization | Stable activations |
| Position encoding | Order-aware without recurrence |
| Cross-attention (enc-dec) | Condition output on input sequence |

---

## 9. Common Interview Questions

Use this section to test your understanding.

**Q1: Why is it called "Attention Is All You Need"?**  
A: The paper shows that attention alone (no RNN, no CNN) is sufficient for state-of-the-art translation.

**Q2: What is the complexity of self-attention?**  
A: O(L² · d) time and memory for sequence length L and dimension d.

**Q3: Why do we need positional encoding?**  
A: Self-attention is permutation-invariant — without position info, "cat sat" and "sat cat" are identical.

**Q4: What is the difference between encoder and decoder self-attention?**  
A: Encoder: all tokens attend to all tokens. Decoder: causal mask — each token attends only to previous tokens.

**Q5: What is cross-attention?**  
A: Queries from decoder, Keys and Values from encoder memory — lets the decoder "look at" the source sentence.

**Q6: What is teacher forcing?**  
A: During training, the decoder receives ground-truth previous tokens as input, not its own predictions.

**Q7: Why Post-LN vs Pre-LN?**  
A: Original paper uses Post-LN. Pre-LN (norm before sublayer) is more stable for very deep models.

**Q8: Could we replace attention with something else?**  
A: Yes — State Space Models (Mamba), linear attention, RWKV — but standard softmax attention remains dominant at scale.

---

## 10. What This Project Does vs. What the Paper Did

| Aspect | Original paper (2017) | This project |
|--------|----------------------|--------------|
| Task | WMT 2014 En→De (millions of sentences) | 8 toy En→Fr pairs |
| d_model | 512 | 128 |
| Layers | 6 encoder + 6 decoder | 2 + 2 |
| Training time | Days on 8 GPUs | ~1-2 minutes on CPU |
| Purpose | Beat SOTA translation | Teach architecture |
| Positional encoding | Sin/cos | Sin/cos (same) |
| Attention | Scaled dot-product multi-head | Same (from scratch) |
| Normalization | Post-LayerNorm | Post-LayerNorm (same) |

**Important:** Our model **overfits** the 8 sentences on purpose. That is a feature — it proves the architecture works. The paper's contribution was showing this architecture **scales** to massive data.

---

## Quick Reference — Paper Section → Project File

| Paper section | Concept | Project file |
|---------------|---------|--------------|
| §3.1 | Encoder-Decoder | `models/encoder.py`, `models/decoder.py`, `models/transformer.py` |
| §3.2 | Scaled dot-product attention | `models/attention.py` → `scaled_dot_product_attention()` |
| §3.2 | Multi-head attention | `models/attention.py` → `MultiHeadAttention` |
| §3.3 | Position-wise FFN | `models/feed_forward.py` |
| §3.4 | Embeddings + linear + softmax | `models/encoder.py`, `models/decoder.py`, `models/transformer.py` |
| §3.5 | Positional encoding | `models/positional_encoding.py` |
| §5 | Training | `train.py`, `config.py` |
| — | Inference / decoding | `inference.py` |
| — | Visualization | `visualizations/`, `demo/showcase.html` |

---

## Next Steps

1. Open `notebooks/01_attention.ipynb` and run every cell with the shape printouts.
2. Read `models/attention.py` side-by-side with Paper Section 3.2.
3. Train the model: `python train.py`
4. Open `demo/showcase.html` and study the attention heatmaps.
5. Read the original PDF: `NIPS-2017-attention-is-all-you-need-Paper.pdf`

> **Mentor tip:** Do not rush to understand everything at once. Master attention first — it is the core idea. Everything else (FFN, LayerNorm, positional encoding) supports that core.

---

*This guide was written for the `transformer-from-scratch` educational project. For setup instructions, see [README.md](README.md).*
