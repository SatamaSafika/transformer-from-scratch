import numpy as np
from .attention import MultiHeadAttention
from .layernorm import LayerNorm

class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        # Feed Forward
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) / np.sqrt(d_model)
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) / np.sqrt(d_ff)
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x, mask):
        # 1. Self Attention
        h = self.ln1.forward(x)
        h, _ = self.attn.forward(h, mask)
        x = x + h

        # 2. Feed Forward
        h2 = self.ln2.forward(x)
        h2 = np.maximum(0, h2 @ self.W1 + self.b1)  # ReLU
        h2 = h2 @ self.W2 + self.b2
        x = x + h2

        return x
