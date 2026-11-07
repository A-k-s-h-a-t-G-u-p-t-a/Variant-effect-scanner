"""Task 2.3: per-residue MSA conservation for every aligned protein -> results/conservation.csv.

Columns: protein, pos, cons (weighted Shannon, high = conserved), jsd (Jensen-Shannon, Capra and
Singh 2007), depth (sequences in the full alignment, before subsampling to 2000).
Resumable: proteins already in the CSV are kept; delete the file to recompute everything.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.conservation import conservation, encode, jsd_conservation, read_a3m
from src.evaluate import DATA, RESULTS

OUT = RESULTS / "conservation.csv"


def main():
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["protein"])
    have = set(done.protein)
    parts = [done] if len(done) else []
    for p in proteins:
        acc, f = p["accession"], DATA / "msa" / f"{p['accession']}.a3m"
        if acc in have or not f.exists():
            continue
        msa = read_a3m(f.read_text())
        assert msa[0] == p["sequence"], f"query mismatch for {acc}"
        X = encode(msa)
        parts.append(pd.DataFrame({"protein": acc, "pos": np.arange(1, len(p["sequence"]) + 1),
                                   "cons": conservation(X), "jsd": jsd_conservation(X),
                                   "depth": len(msa)}))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"{out.protein.nunique()}/{len(proteins)} proteins, {len(out)} residues; "
          f"cons {out.cons.min():.2f}-{out.cons.max():.2f}, jsd {out.jsd.min():.2f}-{out.jsd.max():.2f}")


if __name__ == "__main__":
    main()
