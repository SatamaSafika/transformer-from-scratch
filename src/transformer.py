import numpy as np
from .embeddings import TokenEmbedding, PositionalEncodingSinusoidal
from .decoder_block import DecoderBlock
from .layernorm import LayerNorm
from .utils import softmax

class TransformerDecoder:
    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512, n_layers=6, max_len=512):
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncodingSinusoidal(d_model, max_len)
        self.blocks = [DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.W_out = np.random.randn(d_model, vocab_size).astype(np.float32) / np.sqrt(d_model)
        self.b_out = np.zeros((vocab_size,), dtype=np.float32)

    def _causal_mask(self, seq_len):
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        return mask[np.newaxis, np.newaxis, :, :]

    def forward(self, tokens):
        b, t = tokens.shape
        x = self.embedding.forward(tokens) + self.pos_enc.forward(t)
        mask = self._causal_mask(t)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)
        logits = x @ self.W_out + self.b_out
        probs_last = softmax(logits[:, -1, :], axis=-1)
        return logits, probs_last
