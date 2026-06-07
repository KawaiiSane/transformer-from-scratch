"""
Streamlit interactive demo for the Transformer project.

Usage:
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from dataset import TRANSLATION_PAIRS, Vocabulary
from inference import load_model, translate
from visualizations.attention_heatmaps import plot_attention_heatmap

st.set_page_config(
    page_title="Transformer From Scratch",
    page_icon="🔤",
    layout="wide",
)

st.title("Transformer From Scratch")
st.caption('Educational demo — "Attention Is All You Need" (2017)')


@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None


result = get_model()

with st.sidebar:
    st.header("Settings")
    educational_mode = st.toggle("Educational mode", value=True)
    beam_size = st.slider("Beam size (0 = greedy)", 0, 5, 0)

    st.divider()
    st.markdown("**Example sentences**")
    example = st.selectbox("Pick an example", [p[0] for p in TRANSLATION_PAIRS])

    custom = st.text_input("Or type your own", value=example)

if result is None:
    st.warning("No trained model found. Run `python train.py` first.")
    st.code("cd transformer-from-scratch\npython train.py")
    st.stop()

model, src_vocab, tgt_vocab, device = result

col1, col2 = st.columns(2)
with col1:
    st.subheader("English (input)")
    st.info(custom)
with col2:
    st.subheader("French (output)")
    if st.button("Translate", type="primary"):
        translation, attn = translate(
            model, src_vocab, tgt_vocab, custom, device,
            beam_size=beam_size, return_attention=True,
        )
        st.success(translation)

        if educational_mode:
            st.markdown("**Tensor shapes during inference:**")
            st.markdown(
                "- Source tokens: `(1, src_len)`\n"
                "- Encoder memory: `(1, src_len, d_model)`\n"
                "- Decoder output: `(1, tgt_len, d_model)`\n"
                "- Logits: `(1, tgt_len, tgt_vocab_size)`"
            )

        if attn and attn.get("encoder"):
            st.subheader("Attention Visualizations")
            tab1, tab2, tab3 = st.tabs(["Encoder Self-Attn", "Decoder Cross-Attn", "Training Loss"])

            with tab1:
                enc_w = attn["encoder"][-1][0, 0].cpu().numpy()
                src_tokens = [config.SOS_TOKEN] + custom.split() + [config.EOS_TOKEN]
                n = min(len(src_tokens), enc_w.shape[0])
                fig = plot_attention_heatmap(
                    enc_w[:n, :n], src_tokens[:n], src_tokens[:n],
                    title="Encoder Self-Attention (last layer, head 0)",
                )
                st.pyplot(fig)
                plt.close(fig)
                if educational_mode:
                    st.caption(f"Shape: `(n_heads, src_len, src_len)` → showing head 0: `{enc_w.shape}`")

            with tab2:
                if attn.get("decoder_cross"):
                    # Greedy decode stores per-step cross attention differently
                    st.info("Cross-attention maps are best viewed after running train.py + showcase.html")
                else:
                    st.info("Enable by training and using the HTML showcase for full cross-attention maps.")

            with tab3:
                hist_path = Path(config.TRAINING_HISTORY_PATH)
                if hist_path.exists():
                    with open(hist_path) as f:
                        history = json.load(f)
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(history["train_loss"], label="Train")
                    ax.plot(history["val_loss"], label="Val")
                    ax.legend()
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    st.pyplot(fig)
                    plt.close(fig)

st.divider()
with st.expander("Architecture overview"):
    st.markdown("""
    **Encoder:** Embedding + Positional Encoding → N × (Self-Attention → Add&Norm → FFN → Add&Norm)

    **Decoder:** Embedding + Positional Encoding → N × (Masked Self-Attn → Cross-Attn → FFN)

    **Attention:** `softmax(QK^T / √d_k) · V`
    """)

with st.expander("Toy dataset"):
    st.table([{"English": e, "French": f} for e, f in TRANSLATION_PAIRS])
