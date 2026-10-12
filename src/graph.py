"""Protein similarity: fingerprints, k-nearest-neighbour edges, layout, sequence identity."""
import numpy as np
from Bio import Align
from scipy.stats import rankdata
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from src.score import AAS

# Local (Smith-Waterman) alignment. Global alignment divided by the shorter length inflates identity
# for short sequences (a 50-residue domain scored 38-46% "identical" to unrelated proteins).
_aligner = Align.PairwiseAligner(scoring="blastp", mode="local")


def substitution_fingerprint(logratio: np.ndarray, sequence: str) -> np.ndarray:
    """400-dim 'substitution grammar': for each wild-type -> mutant pair, the mean log-ratio over
    every position holding that wild type. Length-invariant. NaN where the protein lacks that wild type."""
    fp = np.full((20, 20), np.nan)
    wt_idx = np.array([AAS.index(a) for a in sequence])
    for w in range(20):
        rows = wt_idx == w
        if rows.any():
            fp[w] = logratio[rows].mean(axis=0)
    return fp.ravel()


def standardise(X: np.ndarray) -> np.ndarray:
    """Fill missing features with the column mean, drop constant columns (the wild-type diagonal),
    then z-score each column so cosine similarity compares profiles, not overall magnitude."""
    X = X.copy()
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    sd = X.std(axis=0)
    keep = sd > 1e-8
    return (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]


def similarity(X: np.ndarray) -> np.ndarray:
    return cosine_similarity(X)


def knn_edges(S: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    """Undirected edges from each node to its k most similar others. Rank-based, so it works even
    when absolute similarities are tightly bunched (as mean-pooled embeddings are)."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    edges = set()
    for i in range(len(S)):
        for j in np.argsort(S[i])[-k:]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def mds_layout(S: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-D positions by multidimensional scaling (t-SNE is unreliable at n=35).

    Distances are neighbour ranks (how far down each other's similarity list two proteins sit),
    not raw cosine distances: mean-pooled embeddings bunch tightly except for a few outliers, and on
    raw cosine those outliers squeeze everything else into the centre. Ranks match the kNN edges."""
    n = len(S)
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    R = np.vstack([rankdata(-S[i]) for i in range(n)])     # 1 = most similar
    D = (R + R.T) / 2 / n
    np.fill_diagonal(D, 0)
    return MDS(n_components=2, metric="precomputed", init="classical_mds", random_state=seed,
               normalized_stress="auto").fit_transform(D)


def percent_identity(a: str, b: str) -> float:
    """Identical residues as a percentage of the aligned columns of the best local alignment (BLOSUM62).
    On its own this is noisy for short, chance alignments; use alignment_zscore to judge significance."""
    aln = _aligner.align(a, b)[0]
    cols = [(x, y) for x, y in zip(aln[0], aln[1])]
    if not cols:
        return 0.0
    return 100.0 * sum(x == y and x != "-" for x, y in cols) / len(cols)


def alignment_zscore(a: str, b: str, n_shuffles: int = 50, seed: int = 0) -> float:
    """How many standard deviations the local alignment score sits above scores against shuffled
    copies of b (same composition, no order). Above about 8 means a genuine evolutionary relationship."""
    rng = np.random.default_rng(seed)
    score = _aligner.score(a, b)
    shuffled = [_aligner.score(a, "".join(rng.permutation(list(b)))) for _ in range(n_shuffles)]
    return float((score - np.mean(shuffled)) / np.std(shuffled))
