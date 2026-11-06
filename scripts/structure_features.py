"""Task 2.4: AlphaFold structure features for every annotated protein.

Downloads the AlphaFold DB model once per UniProt ID (src/structures.fetch_structure, cached in
data/structures/) and writes results/structure_features.csv with one row per residue:
  plddt     per-residue confidence (0-100, stored as the B-factor column)
  contacts  burial proxy = number of other residues whose C-beta (C-alpha for Gly) lies within 10 A
Residue numbering is checked against our UniProt sequence; mismatches abort the protein.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS
from src.structures import fetch_structure

CUTOFF = 10.0
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def residue_table(pdb_path: Path):
    """{resnum: (aa, cb_or_ca_xyz, plddt)} from ATOM records."""
    res = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name, aa, num = line[12:16].strip(), THREE.get(line[17:20]), int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if num not in res:
            res[num] = {"aa": aa, "ca": None, "cb": None, "plddt": float(line[60:66])}
        if name == "CA":
            res[num]["ca"] = xyz
        elif name == "CB":
            res[num]["cb"] = xyz
    return res


def protein_features(acc: str, seq: str) -> pd.DataFrame | None:
    info = fetch_structure(acc)
    if info is None:
        return None
    res = residue_table(STRUCT_DIR / f"{acc}.pdb")
    nums = sorted(res)
    if nums != list(range(1, len(seq) + 1)) or any(res[i]["aa"] != seq[i - 1] for i in nums):
        print(f"  {acc}: AlphaFold residues do not match the UniProt sequence; skipped")
        return None
    pts = np.stack([res[i]["cb"] if res[i]["cb"] is not None else res[i]["ca"] for i in nums])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    contacts = (d < CUTOFF).sum(axis=1) - 1  # exclude self
    return pd.DataFrame({"protein": acc, "pos": nums, "plddt": [res[i]["plddt"] for i in nums],
                         "contacts": contacts})


STRUCT_DIR = DATA / "structures"


def main():
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    parts, failed = [], []
    for p in proteins:
        df = protein_features(p["accession"], p["sequence"])
        (parts if df is not None else failed).append(df if df is not None else p["accession"])
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULTS / "structure_features.csv", index=False)
    print(f"{len(parts)}/{len(proteins)} proteins, {len(out)} residues; failed: {failed}")
    print(f"plddt range {out.plddt.min():.1f}-{out.plddt.max():.1f}, contacts range "
          f"{out.contacts.min()}-{out.contacts.max()}")


if __name__ == "__main__":
    main()
