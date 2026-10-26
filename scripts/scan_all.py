"""Task 1: scan every qualifying annotated protein (resumable; safe to stop and restart).

Each protein's scan is saved to results/logprobs/<accession>.npy as soon as it finishes, and proteins
that are already cached are skipped. Use --device auto on Colab/Kaggle for a GPU.

Usage: python scripts/scan_all.py [--device cpu|cuda|auto] [--limit N]
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.entropy_sites import fetch_proteins
from src.evaluate import RESULTS, cached_logprobs
from src.model import MODEL_NAME, load_model

ALL_CACHE = "annotated_proteins_all.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--limit", type=int, default=None, help="scan at most N uncached proteins")
    args = ap.parse_args()

    proteins = fetch_proteins(None, ALL_CACHE)
    todo = [p for p in proteins if not (RESULTS / "logprobs" / f"{p['accession']}.npy").exists()]
    print(f"{len(proteins)} proteins, {len(proteins) - len(todo)} cached, {len(todo)} to scan")
    if args.limit is not None:
        todo = todo[:args.limit]
    if not todo:
        return

    tok, model = load_model(device=args.device)
    device = str(next(model.parameters()).device)
    meta = {"model": MODEL_NAME, "device": device, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "protein_list": ALL_CACHE}
    (RESULTS / "scan_meta.json").write_text(json.dumps(meta, indent=1))
    print("scanning on", device)

    for i, p in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p['name']} ({len(p['sequence'])} aa)", flush=True)
        cached_logprobs(tok, model, p["accession"], p["sequence"])  # saves on completion


if __name__ == "__main__":
    main()
