"""Task 1: M-CSA catalytic residues (UniProt numbering) for the annotated proteins.

Source: M-CSA REST API, residue_sequences[].resid (UniProt numbering), checked against our sequence.
A protein with no M-CSA entry gets no M-CSA label (NaN in the table), not all-negative, because
absence from M-CSA is not evidence that a residue is non-catalytic.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA

API = ("https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/"
       "?format=json&entries.proteins.sequences.uniprot_ids={acc}")
THREE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
         "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
         "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
OUT = DATA / "mcsa_labels.json"


def fetch(acc: str):
    with urllib.request.urlopen(API.format(acc=acc), timeout=60) as r:
        data = json.load(r)
    if isinstance(data, dict):  # paginated form
        raise RuntimeError(f"unexpected paginated response for {acc}: keys {list(data)}")
    return data


def main():
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    out, mismatches = {}, []
    for p in proteins:
        acc, seq = p["accession"], p["sequence"]
        pos = set()
        for row in fetch(acc):
            for rs in row.get("residue_sequences", []):
                if rs["uniprot_id"] != acc or not rs.get("is_reference", True):
                    continue
                i, aa = rs["resid"], THREE.get(rs["code"])
                if aa is None or i is None or not (1 <= i <= len(seq)) or seq[i - 1] != aa:
                    mismatches.append((acc, rs["resid"], rs["code"]))
                    continue
                pos.add(i)
        if pos:
            out[acc] = sorted(pos)
        time.sleep(0.2)
    OUT.write_text(json.dumps({"source": "M-CSA REST API", "retrieved": time.strftime("%Y-%m-%d"),
                               "labels": out}, indent=1))
    print(f"{len(out)}/{len(proteins)} proteins have M-CSA entries, "
          f"{sum(map(len, out.values()))} catalytic residues, {len(mismatches)} residue/sequence mismatches")
    for m in mismatches[:20]:
        print("mismatch", m)


if __name__ == "__main__":
    main()
