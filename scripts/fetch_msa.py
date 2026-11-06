"""Task 2.3: one MSA per protein from the public ColabFold MMseqs2 server (uniref + environmental).

Sends the public UniProt sequences to https://api.colabfold.com. Saves data/msa/<acc>.a3m and
records depth in data/msa/depth.csv. Resumable: existing alignments are skipped.

Usage: python scripts/fetch_msa.py [--limit N]
"""
import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.conservation import read_a3m
from src.evaluate import DATA

HOST = "https://api.colabfold.com"
MSA_DIR = DATA / "msa"


def fetch_a3m(seq: str, mode: str = "env") -> str:
    r = requests.post(f"{HOST}/ticket/msa", data={"q": f">query\n{seq}\n", "mode": mode}, timeout=60)
    r.raise_for_status()
    out = r.json()
    while out["status"] in ("UNKNOWN", "RATELIMIT"):
        time.sleep(10)  # polite back-off
        r = requests.post(f"{HOST}/ticket/msa", data={"q": f">query\n{seq}\n", "mode": mode}, timeout=60)
        out = r.json()
    ticket = out["id"]
    while out["status"] in ("PENDING", "RUNNING"):
        time.sleep(5)
        out = requests.get(f"{HOST}/ticket/{ticket}", timeout=60).json()
    if out["status"] != "COMPLETE":
        raise RuntimeError(f"ColabFold server status {out['status']}")
    r = requests.get(f"{HOST}/result/download/{ticket}", timeout=300)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        parts = [tar.extractfile(m).read().decode() for m in tar.getmembers()
                 if m.name.endswith(".a3m")]
    return "\n".join(p.replace("\x00", "") for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    MSA_DIR.mkdir(exist_ok=True)
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    todo = [p for p in proteins if not (MSA_DIR / f"{p['accession']}.a3m").exists()]
    print(f"{len(proteins)} proteins, {len(proteins) - len(todo)} aligned, {len(todo)} to fetch")
    for p in todo[:args.limit]:
        t0 = time.time()
        a3m = fetch_a3m(p["sequence"])
        seqs = read_a3m(a3m)
        assert seqs[0] == p["sequence"], f"query mismatch for {p['accession']}"
        (MSA_DIR / f"{p['accession']}.a3m").write_text(a3m)
        print(f"{p['name']}: {len(seqs)} sequences ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(2)
    rows = []
    for p in proteins:
        f = MSA_DIR / f"{p['accession']}.a3m"
        if f.exists():
            rows.append({"protein": p["accession"], "depth": len(read_a3m(f.read_text()))})
    pd.DataFrame(rows).to_csv(MSA_DIR / "depth.csv", index=False)


if __name__ == "__main__":
    main()
