"""Phase A: stronger conservation baseline (A3), precision-recall metrics (A1), classifier and
feature-transform robustness (A2), Holm correction (A4). Pre-specified in results/analysis_plan.md.

The "best conservation variant" is chosen on the 126 development proteins only, by cross-validated
average precision of the single conservation score (ties broken by AUC), then frozen.
Writes results/phase_a_variants.csv, phase_a_contrasts.csv, phase_a_metrics.csv and
phase_a_best_variant.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cvtools import (SEED, add_transforms, boot_ci, boot_diff, cv_scores, holm, metrics,
                         topk_recall_precision)
from src.evaluate import RESULTS

ENT = "entropy"


def recall_diff_boot(y, s_a, s_b, idx, prots, prot, rng, n_boot=1000):
    """Paired bootstrap of mean recall@top10% (a minus b); duplicates count with multiplicity."""
    ra = {q: topk_recall_precision(y[idx[q]], s_a[idx[q]], prot[idx[q]])[0] for q in prots}
    rb = {q: topk_recall_precision(y[idx[q]], s_b[idx[q]], prot[idx[q]])[0] for q in prots}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(prots, len(prots))
        d.append(np.mean([ra[q] for q in pick]) - np.mean([rb[q] for q in pick]))
    d = np.array(d)
    pt = np.mean(list(ra.values())) - np.mean(list(rb.values()))
    p = max(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))), 1.0 / n_boot)
    return pt, np.percentile(d, 2.5), np.percentile(d, 97.5), p


def main():
    df = pd.read_csv(RESULTS / "site_table.csv")
    var = pd.read_csv(RESULTS / "conservation_variants.csv")
    df = df.merge(var, on=["protein", "pos"], how="left").reset_index(drop=True)
    vnames = [c for c in var.columns if c not in ("protein", "pos")]
    assert df[vnames].notna().all().all()
    # consistency with the original columns
    assert np.allclose(df["shannon_t0.8_n2000"], df.cons) and np.allclose(df["jsd_t0.8_n2000"], df.jsd)
    y = df.label.astype(int).to_numpy()
    prot = df.protein.to_numpy()
    idx = {p: g.index.to_numpy() for p, g in df.groupby("protein")}
    prots = np.array(sorted(idx))
    rng = np.random.default_rng(SEED)
    print(f"{len(df)} residues, {len(prots)} proteins, {y.sum()} sites ({y.mean():.1%})")

    # ---- A3: conservation variants (single feature, grouped CV)
    rows = []
    for v in vnames:
        s = cv_scores(df, [v])
        m = metrics(y, s)
        lo, hi = boot_ci(y, s, idx, prots, "ap", rng)
        rec, prec = topk_recall_precision(y, s, prot)
        rows.append({"variant": v, **m, "ap_lo": lo, "ap_hi": hi,
                     "recall_top10pct": rec, "precision_top10pct": prec})
    vt = pd.DataFrame(rows).sort_values(["ap", "auc"], ascending=False)
    vt.to_csv(RESULTS / "phase_a_variants.csv", index=False)
    print(vt.to_string(index=False, float_format="%.4f"))
    best = vt.iloc[0].variant
    best_auc = vt.sort_values("auc", ascending=False).iloc[0].variant
    print(f"\nbest conservation variant by CV AP: {best} (by AUC: {best_auc})")

    # ---- A1/A2: primary contrast under several classifiers and feature transforms
    t = add_transforms(df, [best, ENT])
    configs = {
        "logreg_raw": ("logreg", lambda f: [f]),
        "logreg_rank": ("logreg", lambda f: [f + "__rank"]),
        "logreg_raw+window": ("logreg", lambda f: [f, f + "__win"]),
        "hgb_raw+window": ("hgb", lambda f: [f, f + "__win"]),
        "hgb_rank+window": ("hgb", lambda f: [f + "__rank", f + "__win"]),
    }
    crow, mrow = [], []
    for cname, (model, expand) in configs.items():
        fc = expand(best)
        s_c = cv_scores(t, fc, model=model)
        s_b = cv_scores(t, fc + expand(ENT), model=model)
        s_e = cv_scores(t, expand(ENT), model=model)
        for lab, s in (("cons", s_c), ("entropy", s_e), ("cons+entropy", s_b)):
            rec, prec = topk_recall_precision(y, s, prot)
            mrow.append({"config": cname, "features": lab, **metrics(y, s),
                         "recall_top10pct": rec, "precision_top10pct": prec})
        for what in ("auc", "ap"):
            pt, lo, hi, p = boot_diff(y, s_b, s_c, idx, prots, what, np.random.default_rng(SEED))
            crow.append({"config": cname, "metric": what, "diff": pt, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
        pt, lo, hi, p = recall_diff_boot(y, s_b, s_c, idx, prots, prot, np.random.default_rng(SEED))
        crow.append({"config": cname, "metric": "recall_top10pct", "diff": pt, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
        print("done", cname, flush=True)
    ct = pd.DataFrame(crow)
    ct["p_holm"] = holm(ct.p_boot.to_numpy())
    ct.insert(0, "contrast", "cons+entropy minus cons")
    ct.insert(0, "conservation_variant", best)
    ct.to_csv(RESULTS / "phase_a_contrasts.csv", index=False)
    mt = pd.DataFrame(mrow)
    mt.insert(0, "conservation_variant", best)
    mt.to_csv(RESULTS / "phase_a_metrics.csv", index=False)
    print("\n", mt.to_string(index=False, float_format="%.4f"))
    print("\n", ct.to_string(index=False, float_format="%.4f"))
    (RESULTS / "phase_a_best_variant.json").write_text(
        json.dumps({"best_variant_by_ap": best, "best_variant_by_auc": best_auc}))


if __name__ == "__main__":
    main()
