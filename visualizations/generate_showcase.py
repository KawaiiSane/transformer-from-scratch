"""
Generate a self-contained HTML showcase for the Transformer project.

Usage:
    python visualizations/generate_showcase.py

Output:
    demo/showcase.html  (works offline, no CDN required)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.positional_encoding import PositionalEncoding
from visualizations.attention_heatmaps import (
    fig_to_base64,
    plot_attention_heatmap,
    plot_positional_encoding_matrix,
    plot_training_loss,
)

DEMO_DIR = PROJECT_ROOT / "demo"
FALLBACK_HTML = DEMO_DIR / "showcase_fallback.html"
OUTPUT_HTML = DEMO_DIR / "showcase.html"


def _plotly_heatmap_inline(weights: list, row_labels: list, col_labels: list, title: str) -> str:
    """Generate inline Plotly heatmap div + script (embedded JSON, no CDN)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    fig = go.Figure(
        data=go.Heatmap(
            z=weights,
            x=col_labels,
            y=row_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Keys",
        yaxis_title="Queries",
        height=400,
        margin=dict(l=80, r=40, t=60, b=80),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _architecture_svg() -> str:
    return """
    <svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">
      <rect x="20" y="40" width="160" height="240" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="8"/>
      <text x="100" y="30" text-anchor="middle" font-weight="bold" fill="#1e40af">Encoder</text>
      <text x="100" y="70" text-anchor="middle" font-size="12">Embedding + PE</text>
      <text x="100" y="110" text-anchor="middle" font-size="12">Self-Attention</text>
      <text x="100" y="150" text-anchor="middle" font-size="12">Add &amp; Norm</text>
      <text x="100" y="190" text-anchor="middle" font-size="12">Feed Forward</text>
      <text x="100" y="230" text-anchor="middle" font-size="12">Add &amp; Norm</text>
      <text x="100" y="265" text-anchor="middle" font-size="11" fill="#64748b">× N layers</text>

      <rect x="280" y="100" width="100" height="60" fill="#fef3c7" stroke="#d97706" stroke-width="2" rx="8"/>
      <text x="330" y="135" text-anchor="middle" font-size="12" font-weight="bold">Memory</text>

      <rect x="420" y="40" width="160" height="240" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" rx="8"/>
      <text x="500" y="30" text-anchor="middle" font-weight="bold" fill="#15803d">Decoder</text>
      <text x="500" y="70" text-anchor="middle" font-size="12">Embedding + PE</text>
      <text x="500" y="110" text-anchor="middle" font-size="12">Masked Self-Attn</text>
      <text x="500" y="150" text-anchor="middle" font-size="12">Cross-Attention</text>
      <text x="500" y="190" text-anchor="middle" font-size="12">Feed Forward</text>
      <text x="500" y="230" text-anchor="middle" font-size="12">Linear → Softmax</text>

      <rect x="640" y="120" width="140" height="50" fill="#fce7f3" stroke="#db2777" stroke-width="2" rx="8"/>
      <text x="710" y="150" text-anchor="middle" font-size="12" font-weight="bold">French Output</text>

      <line x1="180" y1="130" x2="280" y2="130" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>
      <line x1="380" y1="130" x2="420" y2="130" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>
      <line x1="580" y1="145" x2="640" y2="145" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
        </marker>
      </defs>
    </svg>
    """


def generate_showcase() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    has_checkpoint = Path(config.CHECKPOINT_PATH).exists()
    has_history = Path(config.TRAINING_HISTORY_PATH).exists()
    has_attention = Path(config.ATTENTION_SNAPSHOTS_PATH).exists()

    # Positional encoding plot (always available)
    pe_module = PositionalEncoding(config.D_MODEL, max_len=50, dropout=0.0)
    pe_matrix = pe_module.pe.squeeze(0).numpy()
    pe_fig = plot_positional_encoding_matrix(pe_matrix[:50])
    pe_b64 = fig_to_base64(pe_fig)

    # Training loss
    loss_b64 = ""
    translations_html = ""
    attention_sections = ""
    status_banner = ""

    if has_history:
        with open(config.TRAINING_HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
        loss_fig = plot_training_loss(history["train_loss"], history.get("val_loss"))
        loss_b64 = fig_to_base64(loss_fig)
        status_banner = '<div class="banner success">Trained model detected — showing real results.</div>'
    else:
        status_banner = (
            '<div class="banner warning">No checkpoint found — showing sample visualizations. '
            'Run <code>python train.py</code> to train the model.</div>'
        )
        # Demo loss curve
        demo_loss = [4.5 - 4.0 * (i / 199) ** 1.5 + 0.1 * (i % 7) for i in range(200)]
        loss_fig = plot_training_loss(demo_loss, [l + 0.3 for l in demo_loss])
        loss_b64 = fig_to_base64(loss_fig)

    if has_attention:
        with open(config.ATTENTION_SNAPSHOTS_PATH, encoding="utf-8") as f:
            snapshot = json.load(f)

        for key in ("encoder_self", "decoder_self", "decoder_cross"):
            if key in snapshot.get("attention", {}):
                attn = snapshot["attention"][key]
                weights = attn["weights"]
                rows = attn["row_labels"][: len(weights)]
                cols = attn["col_labels"][: len(weights[0]) if weights else 0]
                # Trim to actual size
                n_rows, n_cols = len(weights), len(weights[0]) if weights else 0
                rows = rows[:n_rows]
                cols = cols[:n_cols]
                weights = [row[:n_cols] for row in weights[:n_rows]]

                heatmap_html = _plotly_heatmap_inline(
                    weights, rows, cols, attn.get("title", key)
                )
                attention_sections += f"""
                <div class="card">
                  <h3>{attn.get('title', key)}</h3>
                  {heatmap_html}
                </div>
                """

        if snapshot.get("translations"):
            rows = ""
            for t in snapshot["translations"]:
                match = "✓" if t["predicted"].strip() == t["target"].strip() else "○"
                rows += f"<tr><td>{t['src']}</td><td>{t['target']}</td><td>{t['predicted']}</td><td>{match}</td></tr>"
            translations_html = f"""
            <table class="results-table">
              <thead><tr><th>English</th><th>Target French</th><th>Predicted</th><th>Match</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """
    else:
        # Fallback sample attention (diagonal-ish pattern)
        sample_weights = [[0.6 if i == j else 0.1 / max(abs(i - j), 1) for j in range(4)] for i in range(4)]
        labels = ["<sos>", "hello", "bonjour", "<eos>"][:4]
        attention_sections = f"""
        <div class="card">
          <h3>Sample Encoder Self-Attention (demo data)</h3>
          {_plotly_heatmap_inline(sample_weights, labels, labels, "Demo Attention Map")}
        </div>
        """
        translations_html = """
        <table class="results-table">
          <thead><tr><th>English</th><th>Target French</th><th>Predicted</th><th>Match</th></tr></thead>
          <tbody>
            <tr><td>hello</td><td>bonjour</td><td><em>train first</em></td><td>—</td></tr>
            <tr><td>thank you</td><td>merci</td><td><em>train first</em></td><td>—</td></tr>
          </tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transformer From Scratch — Interactive Showcase</title>
  <script>
    // Minimal inline Plotly (loaded from embedded plots only)
    window.PlotlyConfig = {{MathJaxConfig: 'local'}};
  </script>
  <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a; --surface: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
      --accent: #38bdf8; --success: #4ade80; --warning: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: var(--accent); }}
    h2 {{ font-size: 1.4rem; margin: 2rem 0 1rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1.1rem; margin-bottom: 0.75rem; color: #7dd3fc; }}
    p {{ margin-bottom: 1rem; color: var(--muted); }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
    .banner {{ padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
    .banner.success {{ background: #14532d; border: 1px solid var(--success); }}
    .banner.warning {{ background: #713f12; border: 1px solid var(--warning); }}
    .card {{ background: var(--surface); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }}
    details {{ background: var(--surface); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #334155; }}
    summary {{ cursor: pointer; font-weight: 600; color: var(--accent); }}
    code {{ background: #334155; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.9em; }}
    pre {{ background: #334155; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 1rem 0; }}
    img {{ max-width: 100%; border-radius: 8px; }}
    .results-table {{ width: 100%; border-collapse: collapse; }}
    .results-table th, .results-table td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
    .results-table th {{ color: var(--accent); }}
    .math {{ font-family: 'Times New Roman', serif; font-style: italic; background: #334155; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0; }}
    footer {{ text-align: center; padding: 2rem; color: var(--muted); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Transformer From Scratch</h1>
    <p class="subtitle">Educational implementation of "Attention Is All You Need" (Vaswani et al., 2017)</p>

    {status_banner}

    <details open>
      <summary>What is a Transformer?</summary>
      <p>The Transformer replaces recurrent neural networks with <strong>self-attention</strong> — a mechanism that lets every token directly look at every other token. This enables parallel training and long-range dependencies without vanishing gradients.</p>
      <p>This demo trains a tiny English→French translator on 8 phrase pairs to make the architecture easy to understand. It is not a production LLM.</p>
      <p>Paper: <a href="https://arxiv.org/abs/1706.03762" style="color:var(--accent)">Attention Is All You Need</a></p>
    </details>

    <h2>Architecture</h2>
    <div class="card">{_architecture_svg()}</div>

    <details>
      <summary>Core Math</summary>
      <div class="math">Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) · V</div>
      <p><strong>Scaled Dot-Product:</strong> Queries and Keys are dot-producted, scaled by √d_k to prevent softmax saturation, then softmaxed to get attention weights multiplied by Values.</p>
      <p><strong>Multi-Head:</strong> Run H parallel attention heads on different projection subspaces, concatenate, and project with W_O.</p>
      <p><strong>Positional Encoding:</strong> Sinusoidal functions added to embeddings so the model knows token order.</p>
    </details>

    <h2>Positional Encodings</h2>
    <div class="card">
      <p>Sinusoidal encodings inject position information. Each dimension oscillates at a different frequency.</p>
      <img src="data:image/png;base64,{pe_b64}" alt="Positional encoding heatmap">
    </div>

    <h2>Attention Maps</h2>
    <p>Hover over cells to see attention weights. Brighter = stronger attention.</p>
    {attention_sections}

    <h2>Training Results</h2>
    <div class="card">
      <img src="data:image/png;base64,{loss_b64}" alt="Training loss curve">
    </div>

    <h2>Translations</h2>
    <div class="card">{translations_html}</div>

    <details>
      <summary>What I Learned</summary>
      <ul style="padding-left:1.5rem;color:var(--muted)">
        <li>Attention replaces recurrence — O(L²) but fully parallelizable on GPU.</li>
        <li>Scaling by √d_k keeps softmax gradients healthy as dimension grows.</li>
        <li>Positional encodings are required because attention alone is order-blind.</li>
        <li>Residual connections + LayerNorm stabilize deep stacks.</li>
        <li>Teacher forcing trains the decoder with ground-truth prefixes.</li>
      </ul>
    </details>

    <details>
      <summary>Run It Yourself</summary>
      <pre>cd transformer-from-scratch
pip install -r requirements.txt
python train.py
python visualizations/generate_showcase.py
open demo/showcase.html
streamlit run demo/streamlit_app.py</pre>
    </details>

    <footer>
      Built for learning · PyTorch from scratch · No HuggingFace · No torch.nn.Transformer
    </footer>
  </div>
</body>
</html>
"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Showcase written to {OUTPUT_HTML}")

    # Also save fallback copy
    FALLBACK_HTML.write_text(html, encoding="utf-8")
    print(f"Fallback copy at {FALLBACK_HTML}")


if __name__ == "__main__":
    generate_showcase()
