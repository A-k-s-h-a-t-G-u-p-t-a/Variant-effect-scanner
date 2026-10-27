"""Task 1: cluster the annotated proteins with MMseqs2 (min seq id 0.3), save data/clusters.csv.

Usage: python scripts/cluster.py [path/to/mmseqs.exe]   (default: mmseqs on PATH, then tools/)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA

MIN_SEQ_ID = 0.3
LOCAL = ROOT / "tools" / "mmseqs_dl" / "mmseqs" / "bin" / "mmseqs.exe"


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else (shutil.which("mmseqs") or str(LOCAL))
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    work = ROOT / "tools" / "cluster_work"
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "seqs.fasta"
    fasta.write_text("".join(f">{p['accession']}\n{p['sequence']}\n" for p in proteins))
    # Windows build: workflows run via busybox helpers in the same dir (`busybox.exe --install .` once)
    env = {**os.environ, "PATH": str(Path(exe).resolve().parent) + os.pathsep + os.environ["PATH"]}
    # easy-cluster is a shell script (fails on stock Windows), so run its three steps directly.
    # relative paths + cwd=ROOT: the busybox shell scripts break on the space in "Major Project"
    db, clu, tmp, tsv = (f"tools/cluster_work/{n}" for n in ("db", "clu", "tmp", "out_cluster.tsv"))
    for args in (["createdb", "tools/cluster_work/seqs.fasta", db],
                 ["cluster", db, clu, tmp, "--min-seq-id", str(MIN_SEQ_ID)],
                 ["createtsv", db, db, clu, tsv]):
        subprocess.run([exe, *args, "-v", "1"], check=True, env=env, cwd=ROOT)

    pairs = pd.read_csv(work / "out_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
    ids = {rep: i for i, rep in enumerate(sorted(pairs.rep.unique()))}
    df = pd.DataFrame({"protein": pairs.member, "cluster": pairs.rep.map(ids)}).sort_values("protein")
    assert len(df) == len(proteins) and df.protein.is_unique
    df.to_csv(DATA / "clusters.csv", index=False)
    sizes = df.cluster.value_counts()
    print(f"{len(proteins)} proteins -> {df.cluster.nunique()} clusters at min-seq-id {MIN_SEQ_ID}; "
          f"largest {sizes.max()}, singletons {(sizes == 1).sum()}")


if __name__ == "__main__":
    main()
