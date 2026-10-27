"""Task 0: one tidy per-residue table from cached scans (no model runs).

Columns: protein, pos, aa, label, label_mcsa, entropy, p_wt, mean_logratio, cluster
Rows follow the labelling rules of entropy_sites.py: excluded residues (similarity/prediction-only
annotations) are dropped; label = experimentally supported active/binding site residue.
`p_wt` is the masked probability of the true residue, renormalised over the 20 standard amino acids
(same normalisation as entropy). `cluster` comes from scripts/cluster.py (MMseqs2, 0.3 identity).
`label_mcsa` is the M-CSA catalytic-residue label; empty for proteins with no M-CSA entry.
Only proteins whose scan is cached are included, so the table grows as scans are added.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS, cached_logprobs
from src.score import AAS, entropy_from_logprobs, logratio_from_logprobs

OUT = RESULTS / "site_table.csv"


def protein_rows(p: dict, mcsa: dict, clusters: dict) -> pd.DataFrame:
    seq = p["sequence"]
    lp = cached_logprobs(None, None, p["accession"], seq)  # cache hit required
    prob = np.exp(lp)
    prob /= prob.sum(axis=1, keepdims=True)
    wt_cols = np.array([AAS.index(a) for a in seq])
    p_wt = prob[np.arange(len(seq)), wt_cols]
    mean_lr = logratio_from_logprobs(lp, seq).sum(axis=1) / 19  # WT cell is 0

    keep = np.array([i for i in range(1, len(seq) + 1) if i not in set(p["excluded"])])
    sites = set(p["sites"])
    m = set(mcsa.get(p["accession"], []))
    return pd.DataFrame({
        "protein": p["accession"], "pos": keep, "aa": [seq[i - 1] for i in keep],
        "label": [i in sites for i in keep],
        "label_mcsa": [i in m if m else np.nan for i in keep],
        "entropy": entropy_from_logprobs(lp)[keep - 1],
        "p_wt": p_wt[keep - 1], "mean_logratio": mean_lr[keep - 1], "cluster": clusters.get(p["accession"], np.nan),
    })


def main():
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    proteins = [p for p in proteins if (RESULTS / "logprobs" / f"{p['accession']}.npy").exists()]
    mcsa = json.loads((DATA / "mcsa_labels.json").read_text())["labels"]
    clusters = pd.read_csv(DATA / "clusters.csv").set_index("protein").cluster.to_dict()
    df = pd.concat([protein_rows(p, mcsa, clusters) for p in proteins], ignore_index=True)
    df["cluster"] = df.cluster.astype("Int64")
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows, {df.protein.nunique()} proteins, {int(df.label.sum())} site residues")

    # Task 0 check: the original 25 proteins must reproduce the reported pooled AUCs
    orig = {p["accession"] for p in json.loads((DATA / "annotated_proteins.json").read_text())}
    df = df[df.protein.isin(orig)]
    auc_e = roc_auc_score(df.label, -df.entropy)
    auc_m = roc_auc_score(df.label, -df.mean_logratio)
    print(f"original 25: pooled AUC -entropy {auc_e:.4f}, -mean_logratio {auc_m:.4f}")
    assert round(auc_e, 3) == 0.844 and round(auc_m, 3) == 0.821, "does not reproduce reported AUCs"

    old = pd.read_csv(RESULTS / "entropy_sites_residues.csv")
    assert len(old) == len(df) and (old.pos.values == df.pos.values).all()
    assert np.allclose(old.entropy, df.entropy, atol=1e-5)
    print("matches results/entropy_sites_residues.csv; saved", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
