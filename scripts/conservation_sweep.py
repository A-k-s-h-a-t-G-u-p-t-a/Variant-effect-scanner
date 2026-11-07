"""A3: sweep of conservation-score variants (all frozen names below) -> results/conservation_variants.csv.

Variants differ in sequence-weight identity threshold (theta), pseudocount (lam), alignment
subsample size (n) and window smoothing; `cons`/`jsd` already in site_table.csv are
shannon_t0.8_n2000 / jsd_t0.8_n2000 (checked below). Resumable by protein.

Usage: python scripts/conservation_sweep.py [--msa-dir data/msa --list annotated_proteins_all.json
                                             --out conservation_variants.csv]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.conservation import BG, _freqs, encode, read_a3m
from src.evaluate import DATA, RESULTS

# name: (kind, theta, lam, n_seqs, window_lambda)
VARIANTS = {
    "shannon_t0.8_n2000": ("shannon", 0.8, 1.0, 2000, 0.0),
    "shannon_t0.62_n2000": ("shannon", 0.62, 1.0, 2000, 0.0),
    "shannon_t0.9_n2000": ("shannon", 0.9, 1.0, 2000, 0.0),
    "shannon_t0.8_lam5_n2000": ("shannon", 0.8, 5.0, 2000, 0.0),
    "shannon_t0.8_n500": ("shannon", 0.8, 1.0, 500, 0.0),
    "shannon_t0.8_n100": ("shannon", 0.8, 1.0, 100, 0.0),
    "jsd_t0.8_n2000": ("jsd", 0.8, 1.0, 2000, 0.0),
    "jsd_t0.62_n2000": ("jsd", 0.62, 1.0, 2000, 0.0),
    "jsd_t0.9_n2000": ("jsd", 0.9, 1.0, 2000, 0.0),
    "jsd_t0.8_win0.5_n2000": ("jsd", 0.8, 1.0, 2000, 0.5),
    "jsd_t0.8_n500": ("jsd", 0.8, 1.0, 500, 0.0),
    "jsd_t0.8_n100": ("jsd", 0.8, 1.0, 100, 0.0),
}


def score(kind, F, gap, window_lambda):
    if kind == "shannon":
        return (np.log(20) - (-(F * np.log(F)).sum(1))) * (1 - gap)
    M = 0.5 * (F + BG)
    kl = lambda P, Q: (P * np.log2(P / Q)).sum(1)
    s = (0.5 * kl(F, M) + 0.5 * kl(np.broadcast_to(BG, F.shape), M)) * (1 - gap)
    if window_lambda > 0:
        pad = np.pad(s, 3, mode="edge")
        win = np.array([np.r_[pad[j:j + 3], pad[j + 4:j + 7]].mean() for j in range(len(s))])
        s = (1 - window_lambda) * s + window_lambda * win
    return s


def protein_variants(msa):
    cols, X_by_n, same_by_n = {}, {}, {}
    for name, (kind, theta, lam, n, wl) in VARIANTS.items():
        if n not in X_by_n:
            X_by_n[n] = encode(msa, max_seqs=n)
            X = X_by_n[n]
            same_by_n[n] = np.stack([(X == X[i]).mean(1) for i in range(len(X))])  # once per n
        w = 1.0 / (same_by_n[n] >= theta).sum(1)  # same definition as src.conservation.seq_weights
        F, gap = _freqs(X_by_n[n], w, lam)
        cols[name] = score(kind, F, gap, wl)
    return cols


def work(args):
    acc, seq, path = args
    msa = read_a3m(Path(path).read_text())
    assert msa[0] == seq, acc
    cols = protein_variants(msa)
    return pd.DataFrame({"protein": acc, "pos": np.arange(1, len(seq) + 1), **cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa-dir", default=str(DATA / "msa"))
    ap.add_argument("--list", default="annotated_proteins_all.json")
    ap.add_argument("--out", default="conservation_variants.csv")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    out_path = RESULTS / args.out
    proteins = json.loads((DATA / args.list).read_text())
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["protein"])
    parts, have = ([done] if len(done) else []), set(done.protein)
    jobs = [(p["accession"], p["sequence"], str(Path(args.msa_dir) / f"{p['accession']}.a3m"))
            for p in proteins if p["accession"] not in have and (Path(args.msa_dir) / f"{p['accession']}.a3m").exists()]
    with Pool(args.procs) as pool:
        for i, part in enumerate(pool.imap(work, jobs), 1):
            parts.append(part)
            print(f"[{i}/{len(jobs)}] {part.protein.iloc[0]}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"{out.protein.nunique()} proteins, {len(VARIANTS)} variants -> {out_path}")


if __name__ == "__main__":
    main()
