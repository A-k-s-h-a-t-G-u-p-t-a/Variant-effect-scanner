import numpy as np
import torch

from src.score import TOKEN_OFFSET


@torch.no_grad()
def embed_sequence(tok, model, sequence: str) -> np.ndarray:
    """Mean-pooled final-layer hidden state over real residues only (no <cls>/<eos>)."""
    batch = tok(sequence, return_tensors="pt")
    h = model(**batch, output_hidden_states=True).hidden_states[-1][0]   # (n_tokens, 960)
    residues = h[TOKEN_OFFSET:TOKEN_OFFSET + len(sequence)]
    assert residues.shape[0] == len(sequence), "token span does not match sequence length"
    return residues.mean(dim=0).float().numpy()
