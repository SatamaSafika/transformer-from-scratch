import numpy as np
from src.utils import set_seed
from src.transformer import TransformerDecoder

if __name__ == "__main__":
    print("Starting program...")   # << tambahan debug
    set_seed(123)
    vocab_size, d_model, n_heads, d_ff, n_layers = 50, 32, 4, 128, 2
    model = TransformerDecoder(vocab_size, d_model, n_heads, d_ff, n_layers)
    tokens = np.random.randint(0, vocab_size, size=(2, 6))
    print("Tokens:\n", tokens)      # << tambahan debug
    logits, probs_last = model.forward(tokens)
    print("logits shape:", logits.shape)
    print("probs_last shape:", probs_last.shape)
    print("sum of probs_last:", probs_last.sum(axis=-1))

