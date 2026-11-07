"""Shared helpers for grouped cross-validation and protein-level bootstraps (Phase A onward)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BOOT = 1000
SEED = 0


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    # fixed, untuned settings
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          class_weight="balanced", random_state=0)


def add_transforms(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Adds <f>__rank (percentile rank within protein) and <f>__win (centred +-3 residue mean)."""
    out = df.copy()
    g = out.groupby("protein")
    for f in feats:
        out[f + "__rank"] = g[f].rank(pct=True)
        out[f + "__win"] = g[f].transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
    return out


def cv_scores(df, feats, label="label", model="logreg", n_splits=5):
    """Out-of-fold scores; folds are grouped by `cluster` so homologues never straddle train/test."""
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits).split(df, df[label], groups=df.cluster):
        m = make_model(model)
        m.fit(df.iloc[tr][feats], df[label].iloc[tr])
        oof[te] = m.decision_function(df.iloc[te][feats]) if model == "logreg" \
            else m.predict_proba(df.iloc[te][feats])[:, 1]
    return oof


def metrics(y, s):
    return {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s)}


def topk_recall_precision(y, s, protein, frac=0.10):
    """Mean over proteins of recall and precision among each protein's top `frac` residues."""
    rec, prec = [], []
    for p in np.unique(protein):
        i = np.where(protein == p)[0]
        k = max(1, int(round(frac * len(i))))
        top = i[np.argsort(-s[i])[:k]]
        hit = y[top].sum()
        if y[i].sum():
            rec.append(hit / y[i].sum())
        prec.append(hit / k)
    return float(np.mean(rec)), float(np.mean(prec))


def pooled(y, s, idx, prots, what):
    ii = np.concatenate([idx[p] for p in prots])
    yy = y[ii]
    if not 0 < yy.sum() < len(yy):
        return np.nan
    return roc_auc_score(yy, s[ii]) if what == "auc" else average_precision_score(yy, s[ii])


def boot_diff(y, s_a, s_b, idx, prots, what, rng, n_boot=N_BOOT):
    """Paired protein-level bootstrap of metric(a) - metric(b). Returns point, lo, hi, two-sided p."""
    prots = np.asarray(prots)
    d = np.array([pooled(y, s_a, idx, pick, what) - pooled(y, s_b, idx, pick, what)
                  for pick in (rng.choice(prots, len(prots)) for _ in range(n_boot))])
    pt = pooled(y, s_a, idx, prots, what) - pooled(y, s_b, idx, prots, what)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return pt, lo, hi, max(p, 1.0 / n_boot)


def boot_ci(y, s, idx, prots, what, rng, n_boot=N_BOOT):
    prots = np.asarray(prots)
    v = np.array([pooled(y, s, idx, rng.choice(prots, len(prots)), what) for _ in range(n_boot)])
    return np.nanpercentile(v, [2.5, 97.5])


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
