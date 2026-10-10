"""Control for Phase 3: is the entropy result just amino-acid identity?

Run after scripts/entropy_sites.py. Uses only saved results, no model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]

seqs = {p["accession"]: p["sequence"]
        for p in json.loads((ROOT / "data" / "annotated_proteins.json").read_text())}
df = pd.read_csv(ROOT / "results" / "entropy_sites_residues.csv")
df["aa"] = [seqs[a][p - 1] for a, p in zip(df.accession, df.pos)]

# Baseline: site rate per amino-acid type, learned on held-out proteins (5-fold by protein)
pred = np.zeros(len(df))
for tr, te in GroupKFold(5).split(df, groups=df.accession):
    rate = df.iloc[tr].groupby("aa").site.mean()
    pred[te] = df.iloc[te].aa.map(rate).fillna(0).values

# Within-type AUC: sites vs non-sites of the same amino acid
aucs, weights = [], []
for aa, g in df.groupby("aa"):
    if 0 < g.site.sum() < len(g):
        aucs.append(roc_auc_score(g.site, -g.entropy))
        weights.append(g.site.sum())

print(f"AUC amino-acid identity only (CV): {roc_auc_score(df.site, pred):.3f}")
print(f"AUC -entropy (pooled):             {roc_auc_score(df.site, -df.entropy):.3f}")
print(f"AUC -entropy within same AA type:  {np.average(aucs, weights=weights):.3f}")
print("site residues by amino acid:", df[df.site].aa.value_counts().to_dict())
