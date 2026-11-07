"""C1: confirmatory test on held-out proteins with M-CSA catalytic-residue labels.

Frozen from the development set (126 proteins, UniProt labels): the conservation variant
(results/phase_a_best_variant.json), logistic regression settings, and the fitted models themselves
(trained once on all 126 development proteins; nothing is refit or tuned on C1).
Reports pooled AUC/AP for cons, entropy and cons+entropy on C1, paired protein-level bootstrap of
(cons+entropy) minus cons, and a sensitivity run that drops C1 proteins sharing an MMseqs2 cluster
(30% identity) with any development protein.

Prerequisites: data/msa_c1/*.a3m, results/logprobs_c1/*.npy and
  python scripts/conservation_sweep.py --list c1_proteins.json --msa-dir data/msa_c1 --out c1_conservation_variants.csv
Writes results/c1_results.csv and results/c1_positions.csv.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cvtools import SEED, boot_ci, boot_diff, make_model, metrics
from src.evaluate import DATA, RESULTS, cached_logprobs
from src.score import entropy_from_logprobs

EXE = ROOT / "tools" / "mmseqs_dl" / "mmseqs" / "bin" / "mmseqs.exe"


def homologous_to_dev(c1, dev) -> set[str]:
    """C1 accessions that share an MMseqs2 cluster (min-seq-id 0.3) with a development protein."""
    work = ROOT / "tools" / "cluster_c1"
    work.mkdir(parents=True, exist_ok=True)
    (work / "seqs.fasta").write_text("".join(f">{p['accession']}\n{p['sequence']}\n" for p in dev + c1))
    env = {**os.environ, "PATH": str(EXE.parent) + os.pathsep + os.environ["PATH"]}
    rel = "tools/cluster_c1"
    for args in (["createdb", f"{rel}/seqs.fasta", f"{rel}/db"],
                 ["cluster", f"{rel}/db", f"{rel}/clu", f"{rel}/tmp", "--min-seq-id", "0.3"],
                 ["createtsv", f"{rel}/db", f"{rel}/db", f"{rel}/clu", f"{rel}/out.tsv"]):
        subprocess.run([str(EXE), *args, "-v", "1"], check=True, env=env, cwd=ROOT)
    pairs = pd.read_csv(work / "out.tsv", sep="\t", header=None, names=["rep", "member"])
    devs = {p["accession"] for p in dev}
    shared = set(pairs[pairs.member.isin(devs)].rep)
    return set(pairs[pairs.member.isin({p["accession"] for p in c1}) & pairs.rep.isin(shared)].member)


def main():
    best = json.loads((RESULTS / "phase_a_best_variant.json").read_text())["best_variant_by_ap"]
    dev_df = pd.read_csv(RESULTS / "site_table.csv").merge(
        pd.read_csv(RESULTS / "conservation_variants.csv")[["protein", "pos", best]], on=["protein", "pos"])
    models = {"cons": [best], "entropy": ["entropy"], "cons+entropy": [best, "entropy"]}
    fitted = {k: make_model("logreg").fit(dev_df[f], dev_df.label.astype(int)) for k, f in models.items()}

    c1 = json.loads((DATA / "c1_proteins.json").read_text())
    dev = json.loads((DATA / "annotated_proteins_all.json").read_text())
    cv = pd.read_csv(RESULTS / "c1_conservation_variants.csv")
    rows = []
    for p in c1:
        seq, acc = p["sequence"], p["accession"]
        ent = entropy_from_logprobs(cached_logprobs(None, None, acc, seq, subdir="logprobs_c1"))
        c = cv[cv.protein == acc].set_index("pos")
        keep = np.array([i for i in range(1, len(seq) + 1) if i not in set(p["excluded"])])
        rows.append(pd.DataFrame({"protein": acc, "pos": keep, "label": [int(i in set(p["sites"])) for i in keep],
                                  "entropy": ent[keep - 1], best: c.loc[keep, best].to_numpy()}))
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(RESULTS / "c1_positions.csv", index=False)
    homo = homologous_to_dev(c1, dev)
    print(f"{len(d)} residues, {d.protein.nunique()} proteins, {int(d.label.sum())} M-CSA residues; "
          f"{len(homo)} C1 proteins share a 30%-identity cluster with a development protein: {sorted(homo)}")

    out = []
    for tag, sub in (("all C1", d), ("excluding homologues of dev", d[~d.protein.isin(homo)])):
        sub = sub.reset_index(drop=True)
        y = sub.label.to_numpy()
        idx = {q: g.index.to_numpy() for q, g in sub.groupby("protein")}
        prots = np.array([q for q in sorted(idx) if y[idx[q]].sum() > 0])
        sc = {k: fitted[k].decision_function(sub[f]) for k, f in models.items()}
        for k in models:
            m = metrics(y, sc[k])
            rng = np.random.default_rng(SEED)
            out.append({"set": tag, "model": k, "n_proteins": sub.protein.nunique(), "n_sites": int(y.sum()),
                        "auc": m["auc"], "auc_lo": boot_ci(y, sc[k], idx, prots, "auc", rng)[0],
                        "auc_hi": boot_ci(y, sc[k], idx, prots, "auc", rng)[1], "ap": m["ap"]})
        for what in ("auc", "ap"):
            pt, lo, hi, p = boot_diff(y, sc["cons+entropy"], sc["cons"], idx, prots, what, np.random.default_rng(SEED))
            out.append({"set": tag, "model": f"(cons+entropy) minus cons [{what}]", "n_proteins": sub.protein.nunique(),
                        "n_sites": int(y.sum()), "auc": pt if what == "auc" else np.nan, "ap": pt if what == "ap" else np.nan,
                        "auc_lo": lo if what == "auc" else np.nan, "auc_hi": hi if what == "auc" else np.nan,
                        "p_boot": p, "ci_lo": lo, "ci_hi": hi})
    r = pd.DataFrame(out)
    r.to_csv(RESULTS / "c1_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
