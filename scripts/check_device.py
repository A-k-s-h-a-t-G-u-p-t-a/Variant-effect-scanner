"""Task 1: check that a GPU float32 scan matches the cached CPU scan for one protein.

Rescans a cached protein on --device and compares log-probs, entropy and the ranking of entropy
against results/logprobs/<acc>.npy (made on CPU). Nothing is written. Default protein: CISD1.

Usage: python scripts/check_device.py [--device auto] [--acc Q9NZ45]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS
from src.model import load_model
from src.score import entropy_from_logprobs, scan_logprobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--acc", default="Q9NZ45")
    args = ap.parse_args()

    p = next(x for x in json.loads((DATA / "annotated_proteins_all.json").read_text())
             if x["accession"] == args.acc)
    cpu = np.load(RESULTS / "logprobs" / f"{args.acc}.npy")
    tok, model = load_model(device=args.device)
    dev = next(model.parameters()).device
    new = scan_logprobs(tok, model, p["sequence"])

    d_lp = np.abs(new - cpu).max()
    d_ent = np.abs(entropy_from_logprobs(new) - entropy_from_logprobs(cpu)).max()
    rho = spearmanr(entropy_from_logprobs(new), entropy_from_logprobs(cpu))[0]
    print(f"{args.acc} ({len(p['sequence'])} aa) on {dev} vs cached CPU scan")
    print(f"max |logprob diff| {d_lp:.2e}   max |entropy diff| {d_ent:.2e}   entropy Spearman {rho:.6f}")
    ok = d_lp < 1e-2 and d_ent < 1e-2 and rho > 0.9999
    print("PASS" if ok else "FAIL: investigate before trusting GPU scans")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
