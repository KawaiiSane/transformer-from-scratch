# Transformer From Scratch

An educational implementation of the Transformer architecture from [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017).

This project is **not** a production LLM. It is a clean, heavily commented codebase designed to teach the core ideas of the paper through code, notebooks, and interactive demos.

**New to Transformers?** Start with **[Understand.md](Understand.md)** — a step-by-step guide to the paper, every project file, design alternatives, and real-world use cases.

## What You'll Learn

- Scaled dot-product attention and why we divide by √d_k
- Multi-head attention and the split/merge reshape trick
- Sinusoidal positional encodings
- Encoder-decoder architecture with residual connections and layer normalization
- Teacher forcing, greedy decoding, and attention visualization

## Architecture

```
Source tokens ──► [Embedding + PosEnc] ──► Encoder (×N) ──► Memory
                                                                  │
Target tokens ──► [Embedding + PosEnc] ──► Decoder (×N) ◄─────────┘
                                              │
                                              ▼
                                    Linear → Softmax → French output
```

Each **Encoder layer**: Self-Attention → Add & Norm → Feed-Forward → Add & Norm

Each **Decoder layer**: Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → Feed-Forward → Add & Norm

## Math (Core Equations)

**Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Multi-Head:**
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
```

**Positional Encoding (even/odd dimensions):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Project Structure

```
transformer-from-scratch/
├── config.py              # Hyperparameters
├── dataset.py             # Toy English→French corpus
├── train.py               # Training loop
├── inference.py           # Greedy / beam decoding
├── models/                # Architecture (from scratch)
├── notebooks/             # Step-by-step tutorials
├── visualizations/        # Matplotlib plots + HTML generator
└── demo/                  # showcase.html + Streamlit app
```

## Quick Start

```bash
cd transformer-from-scratch
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Phase 1: explore attention in a notebook
jupyter notebook notebooks/01_attention.ipynb

# Train on the toy dataset (~1-3 min on CPU)
python train.py

# Translate a sentence
python inference.py --src "hello"

# Generate the HTML showcase (works offline)
python visualizations/generate_showcase.py
open demo/showcase.html

# Interactive Streamlit demo
streamlit run demo/streamlit_app.py
```

## Toy Dataset

| English       | French              |
|---------------|---------------------|
| hello         | bonjour             |
| thank you     | merci               |
| good morning  | bonjour             |
| good night    | bonne nuit          |
| how are you   | comment allez vous  |
| see you later | au revoir           |
| i love you    | je t aime           |
| goodbye       | au revoir           |

The dataset is intentionally tiny so the model overfits quickly — proving the architecture works on a laptop.

## Training Results

After ~300 epochs on CPU (~1-2 minutes), the model memorizes the toy corpus:

| English       | Predicted             | Target                | Status |
|---------------|-----------------------|-----------------------|--------|
| hello         | bonjour               | bonjour               | OK     |
| thank you     | merci                 | merci                 | OK     |
| good morning  | bonjour               | bonjour               | OK     |
| good night    | bonjour               | bonne nuit            | Ambiguous* |
| how are you   | comment allez vous    | comment allez vous    | OK     |
| see you later | au revoir             | au revoir             | OK     |
| i love you    | je t aime             | je t aime             | OK     |
| goodbye       | au revoir             | au revoir             | OK     |

\*Both `hello` and `good morning` map to `bonjour`, so the model sometimes generalizes "good night" → `bonjour` instead of `bonne nuit`. This is expected on such a tiny dataset and is useful for discussing attention disambiguation.

**Final train loss:** ~0.0002 | **Interactive showcase:** open `demo/showcase.html` in any browser (works offline after generation).

## What I Learned

- Attention replaces recurrence: every token can directly attend to every other token in O(L²) time.
- Multi-head attention lets the model attend to different representation subspaces in parallel.
- Positional encodings are necessary because self-attention alone is permutation-invariant.
- Layer normalization + residual connections stabilize deep stacks without vanishing gradients.
- Teacher forcing during training exposes the decoder to ground-truth prefixes, avoiding exposure bias during early learning.

## Future Improvements

- [x] Beam search decoding (see `python inference.py --beam 3`)
- [x] Tiny decoder-only GPT (`models/gpt_decoder.py`)
- [ ] RNN seq2seq baseline for comparison
- [ ] Larger dataset (Multi30k subset)

## License

MIT — use freely for learning and teaching.
