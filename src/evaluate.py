from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.score import AAS, logratio_from_logprobs, scan_logprobs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PG_DIR = DATA / "proteingym" / "DMS_ProteinGym_substitutions"


def cached_logprobs(tok, model, key: str, sequence: str, progress=None) -> np.ndarray:
    """Scan once per sequence, reuse forever. Pass tok=model=None to require a cache hit."""
    path = RESULTS / "logprobs" / f"{key}.npy"
    if path.exists():
        lp = np.load(path)
        assert lp.shape == (len(sequence), 20), f"stale cache for {key}"
        return lp
    path.parent.mkdir(parents=True, exist_ok=True)
    lp = scan_logprobs(tok, model, sequence, progress=progress)
    np.save(path, lp)
    return lp


def predict_singles(matrix: np.ndarray, sequence: str, mutants) -> np.ndarray:
    preds = []
    for mutant in mutants:
        wt, pos, mut = mutant[0], int(mutant[1:-1]), mutant[-1]
        assert sequence[pos - 1] == wt, f"WT mismatch for {mutant}"
        preds.append(matrix[pos - 1, AAS.index(mut)])
    return np.array(preds)


def predict_additive(matrix: np.ndarray, sequence: str, mutants) -> np.ndarray:
    """Multi-mutants like 'A24T:G30S' are scored as the sum of their single-mutant scores."""
    return np.array([predict_singles(matrix, sequence, m.split(":")).sum() for m in mutants])


def evaluate_assay(tok, model, dms_id: str, sequence: str) -> dict:
    """`spearman` uses all variants (multi-mutants scored additively), matching ProteinGym's
    published numbers; `spearman_singles` restricts to single mutants."""
    df = pd.read_csv(PG_DIR / f"{dms_id}.csv")
    singles = df[~df["mutant"].str.contains(":")]
    matrix = logratio_from_logprobs(cached_logprobs(tok, model, dms_id, sequence), sequence)
    preds_all = predict_additive(matrix, sequence, df["mutant"])
    preds_single = predict_singles(matrix, sequence, singles["mutant"])
    return {"DMS_id": dms_id, "n_mutants": len(df), "n_singles": len(singles),
            "seq_len": len(sequence),
            "spearman": spearmanr(preds_all, df["DMS_score"]).correlation,
            "spearman_singles": spearmanr(preds_single, singles["DMS_score"]).correlation}
