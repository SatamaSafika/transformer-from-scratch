import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model).astype(np.float32) / np.sqrt(d_model)

    def forward(self, tokens):
        return self.weight[tokens]

class PositionalEncodingSinusoidal:
    def __init__(self, d_model, max_len=512):
        self.pe = self._build_pe(max_len, d_model)

    def _build_pe(self, max_len, d_model):
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def forward(self, seq_len):
        return self.pe[:seq_len][np.newaxis, :, :]
