"""Task 1: pooled AUC with a protein-level bootstrap, plus a one-protein-per-cluster sensitivity run.

Scores are negated (low entropy / low mean log-ratio = site). Resampling is over proteins with
replacement (1000 reps, 2.5-97.5 percentile). Proteins with no M-CSA entry have NaN `label_mcsa`
and are left out of the M-CSA rows. Writes results/auc_bootstrap.csv.

Usage: python scripts/auc_bootstrap.py [--table results/site_table.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import RESULTS

SEED = 0
N_BOOT = 1000
SCORES = {"entropy": "entropy", "mean_logratio": "mean_logratio"}


def pooled_auc(parts, idx, score):
    y = np.concatenate([parts[i][0] for i in idx])
    s = np.concatenate([parts[i][1][score] for i in idx])
    return roc_auc_score(y, -s) if 0 < y.sum() < len(y) else np.nan


def run(df, label, score, rng):
    d = df[df[label].notna()]
    prots = sorted(d.protein.unique())
    parts = {p: (g[label].astype(bool).to_numpy(), g[[score]]) for p, g in d.groupby("protein")}
    parts = {p: (y, {score: g[score].to_numpy()}) for p, (y, g) in parts.items()}
    point = pooled_auc(parts, prots, score)
    boot = np.array([pooled_auc(parts, rng.choice(prots, len(prots)), score) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    clus = d.drop_duplicates("protein").set_index("protein").cluster
    one = np.array([pooled_auc(parts, [rng.choice(grp.index.to_numpy())
                                       for _, grp in clus.groupby(clus)], score)
                    for _ in range(N_BOOT)])
    return {"label": label, "score": score, "n_proteins": len(prots), "n_clusters": clus.nunique(),
            "n_sites": int(d[label].sum()), "auc": point, "ci_lo": lo, "ci_hi": hi,
            "one_per_cluster_mean": np.nanmean(one),
            "one_per_cluster_lo": np.nanpercentile(one, 2.5), "one_per_cluster_hi": np.nanpercentile(one, 97.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(RESULTS / "site_table.csv"))
    df = pd.read_csv(ap.parse_args().table)
    rng = np.random.default_rng(SEED)
    rows = [run(df, lab, sc, rng) for lab in ("label", "label_mcsa") for sc in SCORES]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "auc_bootstrap.csv", index=False)
    print(out.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
