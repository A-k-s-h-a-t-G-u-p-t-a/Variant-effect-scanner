"""Task 3: does PLM entropy add information about functional sites beyond conservation/structure?

Grouped 5-fold CV (groups = MMseqs2 cluster, so homologues never straddle train and test) with a
class-balanced logistic regression on standardised features. Out-of-fold decision scores are pooled
into one AUC per feature set. Intervals are protein-level bootstraps (1000 reps, 2.5-97.5 pct) over
the same out-of-fold scores, so differences are paired. Writes results/task3_cv.csv,
results/task3_diffs.csv, results/task3_auc.png.

Usage: python scripts/task3_cv.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import RESULTS

SEED = 0
N_BOOT = 1000
LABEL = "label"
# direction used for single-feature AUC: +1 high = site, -1 low = site (entropy, log-ratio)
SINGLE = {"cons": 1, "jsd": 1, "entropy": -1, "entropy_esmc_600m": -1, "entropy_esm2_650m": -1,
          "p_wt": 1, "mean_logratio": -1, "min_logratio": -1, "contacts": 1, "plddt": 1}
SETS = {
    "cons": ["cons"],
    "jsd": ["jsd"],
    "entropy": ["entropy"],
    "cons+entropy": ["cons", "entropy"],
    "jsd+entropy": ["jsd", "entropy"],
    "cons+entropy_esm2": ["cons", "entropy_esm2_650m"],
    "cons+entropy_esmc600": ["cons", "entropy_esmc_600m"],
    "cons+all_entropies": ["cons", "entropy", "entropy_esmc_600m", "entropy_esm2_650m"],
    "cons+plddt+contacts": ["cons", "plddt", "contacts"],
    "cons+entropy+plddt+contacts": ["cons", "entropy", "plddt", "contacts"],
    "entropy+plddt+contacts": ["entropy", "plddt", "contacts"],
}
# (name, with-entropy set, baseline set): paired AUC differences with protein-level bootstrap
DIFFS = [("cons+entropy vs cons", "cons+entropy", "cons"),
         ("jsd+entropy vs jsd", "jsd+entropy", "jsd"),
         ("cons+entropy+plddt+contacts vs cons+plddt+contacts", "cons+entropy+plddt+contacts", "cons+plddt+contacts"),
         ("cons+entropy_esm2 vs cons", "cons+entropy_esm2", "cons"),
         ("cons+entropy_esmc600 vs cons", "cons+entropy_esmc600", "cons"),
         ("cons+all_entropies vs cons", "cons+all_entropies", "cons"),
         ("entropy vs cons (single features)", "entropy", "cons")]


def cv_scores(df, feats, n_splits=5):
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits).split(df, df[LABEL], groups=df.cluster):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(df.iloc[tr][feats], df[LABEL].iloc[tr])
        oof[te] = m.decision_function(df.iloc[te][feats])
    return oof


def pooled_auc(y, s, idx_by_protein, prots):
    ii = np.concatenate([idx_by_protein[p] for p in prots])
    yy = y[ii]
    return roc_auc_score(yy, s[ii]) if 0 < yy.sum() < len(yy) else np.nan


def boot_ci(y, scores, idx, prots, rng, names):
    """Bootstrap pooled AUC of each named score vector over resampled proteins (shared resamples)."""
    out = {n: [] for n in names}
    for _ in range(N_BOOT):
        pick = rng.choice(prots, len(prots))
        for n in names:
            out[n].append(pooled_auc(y, scores[n], idx, pick))
    return {n: np.array(v) for n, v in out.items()}


def main():
    df = pd.read_csv(RESULTS / "site_table.csv")
    need = sorted({f for fs in SETS.values() for f in fs} | set(SINGLE))
    assert df[need].notna().all().all(), "feature columns have gaps; rebuild the table first"
    df = df.reset_index(drop=True)
    y = df[LABEL].astype(int).to_numpy()
    idx = {p: g.index.to_numpy() for p, g in df.groupby("protein")}
    prots = np.array(sorted(idx))
    print(f"{len(df)} residues, {len(prots)} proteins, {df.cluster.nunique()} clusters, {y.sum()} sites")

    scores = {f"single:{f}": sg * df[f].to_numpy() for f, sg in SINGLE.items()}
    for name, feats in SETS.items():
        scores[name] = cv_scores(df, feats)

    rng = np.random.default_rng(SEED)
    boots = boot_ci(y, scores, idx, prots, rng, list(scores))
    rows = []
    for n, s in scores.items():
        lo, hi = np.nanpercentile(boots[n], [2.5, 97.5])
        rows.append({"model": n, "auc": roc_auc_score(y, s), "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "task3_cv.csv", index=False)
    print(res.to_string(index=False, float_format="%.3f"))

    def key(n):
        return n if n in scores else f"single:{n}"

    def diff_row(label, a, b, stratum, sub):
        sp = np.array(sorted(set(sub) & set(prots)))
        r = np.random.default_rng(SEED)
        d = []
        for _ in range(N_BOOT):
            pick = r.choice(sp, len(sp))
            d.append(pooled_auc(y, scores[a], idx, pick) - pooled_auc(y, scores[b], idx, pick))
        d = np.array(d)
        pt = pooled_auc(y, scores[a], idx, sp) - pooled_auc(y, scores[b], idx, sp)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {"comparison": label, "stratum": stratum, "n_proteins": len(sp),
                "n_sites": int(sum(y[idx[p]].sum() for p in sp)), "auc_diff": pt, "ci_lo": lo,
                "ci_hi": hi, "p_boot_le_0": float(np.mean(d <= 0))}

    depth = df.drop_duplicates("protein").set_index("protein").depth
    terc = pd.qcut(depth, 3, labels=["shallow", "middle", "deep"])
    strata = {"all": prots} | {f"depth {t} ({int(depth[terc == t].min())}-{int(depth[terc == t].max())})":
                               terc.index[terc == t] for t in ["shallow", "middle", "deep"]}
    drows = []
    for label, a, b in DIFFS:
        for sname, sub in strata.items():
            drows.append(diff_row(label, key(a), key(b), sname, sub))
    dres = pd.DataFrame(drows)
    dres.to_csv(RESULTS / "task3_diffs.csv", index=False)
    print(dres.to_string(index=False, float_format="%.4f"))

    # per-stratum pooled AUC for the main sets
    for sname, sub in strata.items():
        sp = np.array(sorted(set(sub) & set(prots)))
        print(sname, {n: round(pooled_auc(y, scores[n], idx, sp), 3)
                      for n in ("cons", "entropy", "cons+entropy", "cons+entropy+plddt+contacts")})

    # figure: pooled AUC with intervals
    plot = res[res.model.isin(["single:entropy", "single:entropy_esmc_600m", "single:entropy_esm2_650m",
                               "single:jsd", "single:cons", "single:contacts", "entropy", "cons",
                               "cons+entropy", "cons+entropy+plddt+contacts"])].copy()
    plot = plot.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(plot.auc, range(len(plot)), xerr=[plot.auc - plot.ci_lo, plot.ci_hi - plot.auc],
                fmt="o", color="#264653", capsize=3)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels(plot.model)
    ax.set_xlabel("pooled AUC, functional sites (95% protein-level bootstrap)")
    ax.axvline(0.5, color="#bbb", lw=0.8)
    ax.set_xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(RESULTS / "task3_auc.png", dpi=150)
    print("saved results/task3_cv.csv, task3_diffs.csv, task3_auc.png")


if __name__ == "__main__":
    main()
