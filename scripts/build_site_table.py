"""Task 0: one tidy per-residue table from cached scans (no model runs).

Columns: protein, pos, aa, label, label_mcsa, entropy, p_wt, mean_logratio, min_logratio,
entropy_esm2_650m, entropy_esmc_600m (other PLMs, NaN until scanned),
cons, jsd, depth (MSA, scripts/compute_conservation.py), plddt, contacts (AlphaFold, scripts/structure_features.py), cluster
Rows follow the labelling rules of entropy_sites.py: excluded residues (similarity/prediction-only
annotations) are dropped; label = experimentally supported active/binding site residue.
`p_wt` is the masked probability of the true residue, renormalised over the 20 standard amino acids
(same normalisation as entropy). `cluster` comes from scripts/cluster.py (MMseqs2, 0.3 identity).
`min_logratio` is the most damaging of the 19 substitutions (WT cell excluded).
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
OTHER_MODELS = ["esm2_650m", "esmc_600m"]  # entropy_<tag> columns, filled when scans/logprobs_<tag>/ exist


def protein_rows(p: dict, mcsa: dict, clusters: dict) -> pd.DataFrame:
    seq = p["sequence"]
    lp = cached_logprobs(None, None, p["accession"], seq)  # cache hit required
    prob = np.exp(lp)
    prob /= prob.sum(axis=1, keepdims=True)
    wt_cols = np.array([AAS.index(a) for a in seq])
    p_wt = prob[np.arange(len(seq)), wt_cols]
    lr = logratio_from_logprobs(lp, seq)
    mean_lr = lr.sum(axis=1) / 19  # WT cell is 0
    min_lr = np.where(np.arange(20) == wt_cols[:, None], np.inf, lr).min(axis=1)

    keep = np.array([i for i in range(1, len(seq) + 1) if i not in set(p["excluded"])])
    sites = set(p["sites"])
    m = set(mcsa.get(p["accession"], []))
    extra = {}
    for tag in OTHER_MODELS:
        f = RESULTS / f"logprobs_{tag}" / f"{p['accession']}.npy"
        extra[f"entropy_{tag}"] = (entropy_from_logprobs(cached_logprobs(None, None, p["accession"], seq,
                                                                       subdir=f"logprobs_{tag}"))[keep - 1]
                                   if f.exists() else np.nan)
    return pd.DataFrame({
        "protein": p["accession"], "pos": keep, "aa": [seq[i - 1] for i in keep],
        "label": [i in sites for i in keep],
        "label_mcsa": [i in m if m else np.nan for i in keep],
        "entropy": entropy_from_logprobs(lp)[keep - 1],
        "p_wt": p_wt[keep - 1], "mean_logratio": mean_lr[keep - 1],
        "min_logratio": min_lr[keep - 1], **extra, "cluster": clusters.get(p["accession"], np.nan),
    })


def main():
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    proteins = [p for p in proteins if (RESULTS / "logprobs" / f"{p['accession']}.npy").exists()]
    mcsa = json.loads((DATA / "mcsa_labels.json").read_text())["labels"]
    clusters = pd.read_csv(DATA / "clusters.csv").set_index("protein").cluster.to_dict()
    df = pd.concat([protein_rows(p, mcsa, clusters) for p in proteins], ignore_index=True)
    for name in ("conservation", "structure_features"):  # optional per-residue baselines, left-merged
        f = RESULTS / f"{name}.csv"
        if f.exists():
            df = df.merge(pd.read_csv(f), on=["protein", "pos"], how="left")
    cols = [c for c in df.columns if c != "cluster"] + ["cluster"]
    df = df[cols]
    df["cluster"] = df.cluster.astype("Int64")
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows, {df.protein.nunique()} proteins, {int(df.label.sum())} site residues")

    # Task 0 check: the original 25 proteins must reproduce the reported pooled AUCs
    orig = {p["accession"] for p in json.loads((DATA / "annotated_proteins.json").read_text())}
    df = df[df.protein.isin(orig)]
    auc_e = roc_auc_score(df.label, -df.entropy)
    auc_m = roc_auc_score(df.label, -df.mean_logratio)
    print(f"original 25: pooled AUC -entropy {auc_e:.4f}, -mean_logratio {auc_m:.4f}, "
          f"-min_logratio {roc_auc_score(df.label, -df.min_logratio):.4f}, "
          f"p_wt {roc_auc_score(df.label, df.p_wt):.4f}")
    assert round(auc_e, 3) == 0.844 and round(auc_m, 3) == 0.821, "does not reproduce reported AUCs"

    old = pd.read_csv(RESULTS / "entropy_sites_residues.csv")
    assert len(old) == len(df) and (old.pos.values == df.pos.values).all()
    assert np.allclose(old.entropy, df.entropy, atol=1e-5)
    print("matches results/entropy_sites_residues.csv; saved", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
