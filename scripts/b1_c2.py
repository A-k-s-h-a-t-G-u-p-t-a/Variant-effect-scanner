"""B1 (pretraining-familiarity strata) and C2 (active-site vs binding-site labels).

Frozen settings from Phase A: best conservation variant (results/phase_a_best_variant.json), logistic
regression on raw features, grouped 5-fold CV by cluster. Contrast = (cons+entropy) minus cons.

B1 proxies (per protein, terciles): model familiarity (mean log p of the true residue under ESMC-300M,
i.e. negative pseudo-log-perplexity), number of alignment sequences with >= 90% identity to the
query, UniProt annotation score and number of PDB entries (UniProt query; skipped if unavailable).
C2: positives = experimentally supported active-site vs binding-site residues (same evidence rules),
residues of the other type are left out of that analysis. Writes results/b1_strata.csv, c2_labels.csv
and data/site_types.json (cache).
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.entropy_sites import EXP_CODES, MAX_SPAN
from src.conservation import read_a3m
from src.cvtools import SEED, boot_diff, cv_scores
from src.evaluate import DATA, RESULTS, cached_logprobs
from src.score import AAS


def uniprot_info(accs):
    """annotation score, PDB count and active/binding residues (experimental, span <= 5) per accession."""
    cache = DATA / "site_types.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for i in range(0, len(accs), 50):
        q = " OR ".join(f"accession:{a}" for a in accs[i:i + 50])
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": q, "fields": "accession,annotation_score,xref_pdb,ft_act_site,ft_binding",
             "format": "json", "size": 100})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    res = json.load(r)["results"]
                break
            except OSError as e:
                print("  UniProt retry", attempt + 1, type(e).__name__, flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError("UniProt unavailable")
        for e in res:
            act, bind = set(), set()
            for f in e.get("features", []):
                s, t = f["location"]["start"]["value"], f["location"]["end"]["value"]
                codes = {ev.get("evidenceCode") for ev in f.get("evidences", [])}
                if s and t and t - s + 1 <= MAX_SPAN and codes & EXP_CODES:
                    (act if f["type"] == "Active site" else bind if f["type"] == "Binding site" else set()).update(range(s, t + 1))
            pdbs = [x for x in e.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"]
            out[e["primaryAccession"]] = {"annotation_score": e.get("annotationScore"), "n_pdb": len(pdbs),
                                          "active": sorted(act), "binding": sorted(bind)}
        time.sleep(1)
    cache.write_text(json.dumps(out))
    return out


def main():
    best = json.loads((RESULTS / "phase_a_best_variant.json").read_text())["best_variant_by_ap"]
    df = pd.read_csv(RESULTS / "site_table.csv")
    var = pd.read_csv(RESULTS / "conservation_variants.csv")[["protein", "pos", best]]
    df = df.merge(var, on=["protein", "pos"]).reset_index(drop=True)
    y = df.label.astype(int).to_numpy()
    idx = {p: g.index.to_numpy() for p, g in df.groupby("protein")}
    prots = np.array(sorted(idx))
    s_c = cv_scores(df, [best])
    s_b = cv_scores(df, [best, "entropy"])

    # ---- B1 proxies
    seqs = {p["accession"]: p["sequence"] for p in json.loads((DATA / "annotated_proteins_all.json").read_text())}
    fam, close = {}, {}
    for acc, seq in seqs.items():
        lp = cached_logprobs(None, None, acc, seq)
        prob = np.exp(lp)
        prob /= prob.sum(1, keepdims=True)
        fam[acc] = float(np.log(prob[np.arange(len(seq)), [AAS.index(a) for a in seq]]).mean())
        msa = read_a3m((DATA / "msa" / f"{acc}.a3m").read_text())
        q = np.frombuffer(msa[0].encode(), dtype=np.uint8)
        ident = [(np.frombuffer(m.encode(), dtype=np.uint8) == q).mean() for m in msa[1:]]
        close[acc] = int(np.sum(np.array(ident) >= 0.9))
    info = uniprot_info(sorted(seqs))
    prox = pd.DataFrame({"familiarity_mean_logp": fam, "n_close_homologs_ge90": close,
                         "annotation_score": {a: info[a]["annotation_score"] for a in info},
                         "n_pdb": {a: info[a]["n_pdb"] for a in info}})
    prox.to_csv(RESULTS / "b1_proxies.csv")

    rows = []
    for name in prox.columns:
        v = prox[name].reindex(prots)
        ranks = v.rank(method="first")
        terc = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
        for t in ["low", "mid", "high"]:
            sub = terc.index[terc == t].to_numpy()
            for what in ("auc", "ap"):
                pt, lo, hi, p = boot_diff(y, s_b, s_c, idx, sub, what, np.random.default_rng(SEED))
                rows.append({"proxy": name, "tercile": t, "range": f"{v[sub].min():.3g} to {v[sub].max():.3g}",
                             "n_proteins": len(sub), "metric": what, "diff": pt, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
    b1 = pd.DataFrame(rows)
    b1.to_csv(RESULTS / "b1_strata.csv", index=False)
    print(b1.to_string(index=False, float_format="%.4f"))
    print("\nproxy correlations (Spearman):\n", prox.corr(method="spearman").round(2))

    # ---- C2: label types
    rows = []
    for lab in ("active", "binding"):
        other = "binding" if lab == "active" else "active"
        pos_set = {(a, p) for a, d in info.items() for p in d[lab]}
        oth_set = {(a, p) for a, d in info.items() for p in d[other]}
        key = list(zip(df.protein, df.pos))
        yl = np.array([k in pos_set for k in key], dtype=int)
        keep = np.array([(k not in oth_set) or (k in pos_set) for k in key])  # drop other-type-only residues
        d2 = df[keep].reset_index(drop=True).assign(lab=yl[keep])
        i2 = {p: g.index.to_numpy() for p, g in d2.groupby("protein")}
        p2 = np.array([p for p in sorted(i2) if d2.lab.to_numpy()[i2[p]].sum() > 0])
        sc = cv_scores(d2, [best], label="lab")
        sb = cv_scores(d2, [best, "entropy"], label="lab")
        se = cv_scores(d2, ["entropy"], label="lab")
        for what in ("auc", "ap"):
            pt, lo, hi, p = boot_diff(d2.lab.to_numpy(), sb, sc, i2, p2, what, np.random.default_rng(SEED))
            from src.cvtools import metrics
            rows.append({"label_type": lab, "n_proteins_with_sites": len(p2), "n_sites": int(d2.lab.sum()),
                         "metric": what, "cons": metrics(d2.lab.to_numpy(), sc)[what],
                         "entropy": metrics(d2.lab.to_numpy(), se)[what],
                         "cons+entropy": metrics(d2.lab.to_numpy(), sb)[what],
                         "diff": pt, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
    c2 = pd.DataFrame(rows)
    c2.to_csv(RESULTS / "c2_labels.csv", index=False)
    print("\n", c2.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
