import numpy as np
from .utils import softmax

class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: shape (B*H, T, d_k)
        """
        # QK^T -> (B*H, T, T)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + (mask * -1e9)

        attn = softmax(scores, axis=-1)  # (B*H, T, T)
        out = attn @ V                   # (B*H, T, d_k)

        print(f"[DEBUG] ScaledDotProductAttention "
              f"Q: {Q.shape} K: {K.shape} V: {V.shape} "
              f"scores: {scores.shape} attn: {attn.shape} out: {out.shape}")

        return out, attn


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model harus habis dibagi n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # inisialisasi matriks bobot
        limit = 1 / np.sqrt(d_model)
        self.W_q = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_k = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_v = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))

        self.attn = ScaledDotProductAttention(self.d_k)

    def split_heads(self, x, b, t):
        """
        x: (b, t, d_model)
        return: (b, h, t, d_k)
        """
        return x.reshape(b, t, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        b, t, d_model = x.shape

        # Linear projection
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Split heads -> (b, h, t, d_k)
        Q = self.split_heads(Q, b, t)
        K = self.split_heads(K, b, t)
        V = self.split_heads(V, b, t)

        # Flatten batch & heads -> (b*h, t, d_k)
        Q = Q.reshape(b * self.n_heads, t, self.d_k)
        K = K.reshape(b * self.n_heads, t, self.d_k)
        V = V.reshape(b * self.n_heads, t, self.d_k)

        # Attention
        out, attn = self.attn.forward(Q, K, V, mask)   # (b*h, t, d_k)

        # Reshape back -> (b, t, d_model)
        out = out.reshape(b, self.n_heads, t, self.d_k).transpose(0, 2, 1, 3)
        out = out.reshape(b, t, d_model)

        # Output projection
        out = out @ self.W_o

        print(f"[DEBUG] MultiHeadAttention out final: {out.shape}")
        return out, attn