"""MSA conservation scores (high = conserved). Query is the first sequence of the alignment."""
import re

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
# Background amino-acid frequencies (BLOSUM62 background, Henikoff and Henikoff 1992), order = AA
BG = np.array([0.074, 0.025, 0.054, 0.054, 0.047, 0.074, 0.026, 0.068, 0.058, 0.099,
               0.025, 0.045, 0.039, 0.034, 0.052, 0.057, 0.051, 0.073, 0.013, 0.032])
BG = BG / BG.sum()
MAX_SEQS = 2000


def read_a3m(text: str) -> list[str]:
    """Sequences from an a3m string with lowercase insertions removed, so columns match the query."""
    seqs, cur = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        elif line.strip() and not line.startswith("#"):
            cur.append(re.sub("[a-z]", "", line.strip()))
    if cur:
        seqs.append("".join(cur))
    L = len(seqs[0])
    return [s for s in seqs if len(s) == L]


def encode(msa: list[str], max_seqs: int = MAX_SEQS, seed: int = 0) -> np.ndarray:
    """(N, L) ints, gap/unknown = -1. Deep alignments are subsampled (query kept) for speed."""
    if len(msa) > max_seqs:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(np.arange(1, len(msa)), max_seqs - 1, replace=False))
        msa = [msa[0]] + [msa[i] for i in keep]
    return np.array([[AA.find(c) for c in s] for s in msa])


def seq_weights(X: np.ndarray, theta: float = 0.8) -> np.ndarray:
    same = np.stack([(X == X[i]).mean(1) for i in range(len(X))])
    return 1.0 / (same >= theta).sum(1)


def _freqs(X, w, lam):
    """(L, 20) pseudocount-smoothed weighted frequencies and (L,) gap fractions."""
    N, L = X.shape
    F = np.zeros((L, 20))
    gap = np.zeros(L)
    for j in range(L):
        col = X[:, j]
        ok = col >= 0
        f = np.bincount(col[ok], weights=w[ok], minlength=20)
        gap[j] = 1 - w[ok].sum() / w.sum()
        F[j] = (f + lam / 20) / (f.sum() + lam)
    return F, gap


def conservation(msa, theta=0.8, lam=1.0) -> np.ndarray:
    """Weighted Shannon conservation (ln 20 - H) x (1 - gap fraction)."""
    X = msa if isinstance(msa, np.ndarray) else encode(msa)
    F, gap = _freqs(X, seq_weights(X, theta), lam)
    H = -(F * np.log(F)).sum(1)
    return (np.log(20) - H) * (1 - gap)


def jsd_conservation(msa, theta=0.8, lam=1.0, window_lambda=0.0) -> np.ndarray:
    """Jensen-Shannon divergence from the background distribution (Capra and Singh 2007),
    gap-penalised. window_lambda > 0 mixes in the mean of the +-3 neighbouring columns."""
    X = msa if isinstance(msa, np.ndarray) else encode(msa)
    F, gap = _freqs(X, seq_weights(X, theta), lam)
    M = 0.5 * (F + BG)
    kl = lambda P, Q: (P * np.log2(P / Q)).sum(1)
    jsd = (0.5 * kl(F, M) + 0.5 * kl(np.broadcast_to(BG, F.shape), M)) * (1 - gap)
    if window_lambda > 0:
        pad = np.pad(jsd, 3, mode="edge")
        win = np.array([np.r_[pad[j:j + 3], pad[j + 4:j + 7]].mean() for j in range(len(jsd))])
        jsd = (1 - window_lambda) * jsd + window_lambda * win
    return jsd
