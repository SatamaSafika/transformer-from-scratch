import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) / np.sqrt(d_model)
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) / np.sqrt(d_ff)
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x):
        return self.gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2