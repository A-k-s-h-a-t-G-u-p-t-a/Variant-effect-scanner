"""Gate 2 + 3: sanity-check known HBB mutations, run the full scan, save heatmaps.

Usage: python scripts/scan.py [UNIPROT_ID]   (default P68871, human HBB)
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import load_model
from src.plot import plot_scan
from src.score import AAS, entropy_from_logprobs, logratio_from_logprobs, scan_logprobs

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def fetch_uniprot(uid: str) -> str:
    cache = ROOT / "data" / f"{uid}.fasta"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta") as r:
            cache.write_bytes(r.read())
    lines = cache.read_text().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def lookup(matrix, seq, mutant):
    wt, pos, mut = mutant[0], int(mutant[1:-1]), mutant[-1]
    assert seq[pos - 1] == wt, f"WT mismatch for {mutant}: sequence has {seq[pos - 1]}"
    return float(matrix[pos - 1, AAS.index(mut)])


def main(uid="P68871"):
    seq = fetch_uniprot(uid)
    print(f"{uid}: {len(seq)} residues")

    lp_path = RESULTS / f"logprobs_{uid}.npy"
    if lp_path.exists():
        logprobs = np.load(lp_path)
    else:
        tok, model = load_model()
        logprobs = scan_logprobs(tok, model, seq)
        np.save(lp_path, logprobs)

    matrix = logratio_from_logprobs(logprobs, seq)
    np.save(RESULTS / f"scan_{uid}.npy", matrix)
    np.save(RESULTS / f"entropy_{uid}.npy", entropy_from_logprobs(logprobs))

    wt_cells = matrix[np.arange(len(seq)), [AAS.index(a) for a in seq]]
    assert np.allclose(wt_cells, 0), "WT cells must be exactly 0"

    if uid == "P68871":
        # UniProt numbering includes Met1, so clinical E6V (sickle cell) = E7V here.
        checks = {
            "E7V   sickle cell (clinical E6V)": "E7V",
            "H93Y  proximal heme His (Hb M Hyde Park)": "H93Y",
            "H64Y  distal heme His (Hb M Boston)": "H64Y",
            "F42S  heme pocket (Hb Hammersmith)": "F42S",
            "K9R   conservative, surface": "K9R",
            "E8D   conservative, surface": "E8D",
            "V2I   conservative, N-term": "V2I",
        }
        print("\nGate 2 sanity check (more negative = more damaging):")
        for label, m in checks.items():
            print(f"  {lookup(matrix, seq, m):+7.2f}  {label}")
        nonwt = matrix[matrix != 0]
        print(f"  {np.median(nonwt):+7.2f}  median of all {nonwt.size} substitutions")

    for start in range(1, len(seq) + 1, 60):
        end = min(start + 59, len(seq))
        out = RESULTS / f"heatmap_{uid}_{start}-{end}.png"
        plot_scan(matrix, seq, out, start, end, title=f"{uid} positions {start}-{end}")
        print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main(*sys.argv[1:])
