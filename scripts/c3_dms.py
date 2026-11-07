"""C3: annotation-free check on 9 ProteinGym assays (rules fixed in results/analysis_plan.md).

Position label: "sensitive" if >= 50% of its measured single substitutions are binarised deleterious
(DMS_score_bin == 0); positions with < 5 measured substitutions are dropped.
Features: ESMC-300M entropy (cached scans), frozen best conservation variant (Phase A) and the
default `cons`, from ColabFold alignments in data/msa_c3/. Leave-one-assay-out logistic regression.
Writes results/c3_positions.csv and results/c3_results.csv.

Usage: python scripts/c3_dms.py   (run scripts/conservation_sweep.py --list c3_proteins.json
                                   --msa-dir data/msa_c3 --out c3_conservation_variants.csv first)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, PG_DIR, RESULTS, cached_logprobs
from src.score import entropy_from_logprobs

SEED = 0
MIN_SUBS = 5
FRAC = 0.5


def positions(dms_id: str, seq: str) -> pd.DataFrame:
    df = pd.read_csv(PG_DIR / f"{dms_id}.csv")
    df = df[~df.mutant.str.contains(":")].copy()
    df["pos"] = df.mutant.str[1:-1].astype(int)
    assert all(seq[p - 1] == m[0] for p, m in zip(df.pos, df.mutant)), f"WT mismatch in {dms_id}"
    g = df.groupby("pos")
    out = pd.DataFrame({"n_subs": g.size(), "frac_del": g.DMS_score_bin.apply(lambda s: (s == 0).mean()),
                        "mean_score": g.DMS_score.mean()}).reset_index()
    out = out[out.n_subs >= MIN_SUBS]
    out["label"] = (out.frac_del >= FRAC).astype(int)
    return out


def auc_ap(y, s):
    return (roc_auc_score(y, s), average_precision_score(y, s)) if 0 < y.sum() < len(y) else (np.nan, np.nan)


def main():
    best = json.loads((RESULTS / "phase_a_best_variant.json").read_text())["best_variant_by_ap"]
    prots = json.loads((DATA / "c3_proteins.json").read_text())
    cv = pd.read_csv(RESULTS / "c3_conservation_variants.csv")
    rows = []
    for p in prots:
        pos = positions(p["accession"], p["sequence"])
        ent = entropy_from_logprobs(cached_logprobs(None, None, p["accession"], p["sequence"]))
        c = cv[cv.protein == p["accession"]].set_index("pos")
        pos["protein"] = p["accession"]
        pos["entropy"] = ent[pos.pos - 1]
        pos["cons"] = c.loc[pos.pos, "shannon_t0.8_n2000"].to_numpy()
        pos["cons_best"] = c.loc[pos.pos, best].to_numpy()
        rows.append(pos)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(RESULTS / "c3_positions.csv", index=False)
    print(f"{len(d)} positions, {d.protein.nunique()} assays, {d.label.mean():.0%} sensitive; best variant {best}")

    # leave-one-assay-out
    sets = {"cons": ["cons_best"], "entropy": ["entropy"], "cons+entropy": ["cons_best", "entropy"]}
    oof = {k: np.zeros(len(d)) for k in sets}
    for a in d.protein.unique():
        te = (d.protein == a).to_numpy()
        for k, f in sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            m.fit(d.loc[~te, f], d.label[~te])
            oof[k][te] = m.decision_function(d.loc[te, f])

    res = []
    for a, g in d.groupby("protein"):
        i = g.index.to_numpy()
        row = {"assay": a, "n_pos": len(g), "frac_sensitive": g.label.mean()}
        for name, s in [("-entropy", -g.entropy), ("cons", g.cons), ("cons_best", g.cons_best)] + \
                       [(k, pd.Series(oof[k][i], index=i)) for k in sets]:
            row[f"auc_{name}"], row[f"ap_{name}"] = auc_ap(g.label.to_numpy(), s.to_numpy())
        # continuous: association of conservation/entropy with mean DMS effect (higher score = fitter)
        row["spearman_-entropy_vs_(1-mean_score)"] = pd.Series(-g.entropy).corr(1 - g.mean_score, method="spearman")
        row["spearman_cons_vs_(1-mean_score)"] = g.cons_best.corr(1 - g.mean_score, method="spearman")
        res.append(row)
    r = pd.DataFrame(res)
    r["auc_gain_(cons+entropy)-cons"] = r["auc_cons+entropy"] - r["auc_cons"]
    r["ap_gain_(cons+entropy)-cons"] = r["ap_cons+entropy"] - r["ap_cons"]
    r.to_csv(RESULTS / "c3_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(r[["assay", "n_pos", "frac_sensitive", "auc_-entropy", "auc_cons_best", "auc_cons+entropy",
             "auc_gain_(cons+entropy)-cons", "ap_gain_(cons+entropy)-cons"]].to_string(index=False, float_format="%.3f"))
    print("\nmean per-assay:", r[["auc_-entropy", "auc_cons_best", "auc_cons+entropy",
                                  "auc_gain_(cons+entropy)-cons", "ap_gain_(cons+entropy)-cons"]].mean().round(4).to_dict())
    rng = np.random.default_rng(SEED)
    for col in ("auc_gain_(cons+entropy)-cons", "ap_gain_(cons+entropy)-cons"):
        v = r[col].to_numpy()
        boot = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        print(f"{col}: mean {v.mean():+.4f}, bootstrap over assays 95% [{np.percentile(boot, 2.5):+.4f}, "
              f"{np.percentile(boot, 97.5):+.4f}], assays with gain > 0: {(v > 0).sum()}/{len(v)}")
    pooled = {k: auc_ap(d.label.to_numpy(), v) for k, v in oof.items()}
    print("pooled (leave-one-assay-out) AUC/AP:", {k: tuple(round(x, 3) for x in v) for k, v in pooled.items()})


if __name__ == "__main__":
    main()
