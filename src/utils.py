import numpy as np
import random

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
