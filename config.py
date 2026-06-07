"""
Hyperparameters for the educational Transformer.

These values are intentionally small so training finishes in 1-3 minutes on CPU.
They follow the ratios from "Attention Is All You Need" (Vaswani et al., 2017)
but scaled down for a toy English→French dataset.
"""

# Reproducibility
SEED = 42

# Model architecture (Paper Section 3)
D_MODEL = 128          # Embedding / hidden dimension
N_HEADS = 4            # Number of attention heads (d_k = d_v = D_MODEL // N_HEADS = 32)
D_FF = 512             # Feed-forward inner dimension (4 × d_model in the paper)
N_LAYERS = 2           # Encoder and decoder stack depth
DROPOUT = 0.0          # Disabled for tiny dataset (overfitting is expected/intentional)
MAX_SEQ_LEN = 20       # Maximum sequence length (toy phrases are short)

# Training
BATCH_SIZE = 8
EPOCHS = 300
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0

# Paths
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "checkpoints/vocab.json"
TRAINING_HISTORY_PATH = "checkpoints/training_history.json"
ATTENTION_SNAPSHOTS_PATH = "checkpoints/attention_snapshots.json"

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Derived (for convenience)
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
