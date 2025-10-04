import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((d_model,), dtype=np.float32)  # scale
        self.beta = np.zeros((d_model,), dtype=np.float32)  # shift

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        norm_x = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta
