"""C1 (held-out confirmatory set): reviewed human proteins, 100-350 aa, standard residues, with M-CSA
catalytic residues, not among the 126 development proteins.

Rule (no hand-picking): sort by accession, shuffle with seed 0, keep the first N that have at least
one M-CSA residue whose UniProt numbering matches our sequence. Writes data/c1_proteins.json with the
M-CSA catalytic residues as positives and `excluded` = residues with experimentally supported UniProt
active/binding-site annotations (not positives under this label set, and not negatives either).

Usage: python scripts/select_c1.py [--n 100]
"""
import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.entropy_sites import EXP_CODES, MAX_SPAN, SITE_TYPES
from scripts.fetch_mcsa import THREE, fetch
from src.evaluate import DATA
from src.score import AAS

CSV_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/media/flat_files/curated_data.csv"
SEED = 0


def retry(fn, *a, tries=4):
    for k in range(tries):
        try:
            return fn(*a)
        except (OSError, ValueError) as e:  # timeouts, resets, bad JSON
            print(f"  retry {k + 1} after {type(e).__name__}", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("giving up after retries")


def mcsa_accessions() -> set[str]:
    with urllib.request.urlopen(CSV_URL, timeout=120) as r:
        text = r.read().decode()
    ids = set()
    for row in csv.DictReader(io.StringIO(text)):
        for u in row["Uniprot IDs"].replace(";", ",").replace(":", ",").split(","):
            if u.strip():
                ids.add(u.strip())
    return ids


def uniprot_entries(accs: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(accs), 60):
        q = "reviewed:true AND organism_id:9606 AND length:[100 TO 350] AND (" + \
            " OR ".join(f"accession:{a}" for a in accs[i:i + 60]) + ")"
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": q, "fields": "accession,id,sequence,ft_act_site,ft_binding", "format": "json", "size": 500})
        with urllib.request.urlopen(url, timeout=120) as r:
            out += json.load(r)["results"]
        time.sleep(0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    n = ap.parse_args().n

    dev = {p["accession"] for p in json.loads((DATA / "annotated_proteins_all.json").read_text())}
    ids = sorted(retry(mcsa_accessions))
    print(f"{len(ids)} UniProt IDs in M-CSA")
    entries = [e for e in retry(uniprot_entries, ids) if e["primaryAccession"] not in dev
               and not set(e["sequence"]["value"]) - set(AAS)]
    entries.sort(key=lambda e: e["primaryAccession"])
    random.Random(SEED).shuffle(entries)
    print(f"{len(entries)} candidate proteins after the length/organism/standard-residue filter")

    out = []
    for e in entries:
        acc, seq = e["primaryAccession"], e["sequence"]["value"]
        pos = set()
        for row in retry(fetch, acc):
            for rs in row.get("residue_sequences", []):
                i = rs["resid"]
                if rs["uniprot_id"] == acc and rs.get("is_reference", True) and i and 1 <= i <= len(seq) \
                        and THREE.get(rs["code"]) == seq[i - 1]:
                    pos.add(i)
        time.sleep(0.2)
        if not pos:
            continue
        uni = set()
        for f in e.get("features", []):
            s, t = f["location"]["start"]["value"], f["location"]["end"]["value"]
            codes = {ev.get("evidenceCode") for ev in f.get("evidences", [])}
            if f["type"] in SITE_TYPES and s and t and t - s + 1 <= MAX_SPAN and codes & EXP_CODES:
                uni.update(range(s, t + 1))
        out.append({"accession": acc, "name": e["uniProtkbId"], "sequence": seq,
                    "sites": sorted(pos), "excluded": sorted(uni - pos)})
        if len(out) == n:
            break
    (DATA / "c1_proteins.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} proteins, {sum(len(p['sites']) for p in out)} M-CSA residues, "
          f"{sum(len(p['sequence']) for p in out)} residues in total")


if __name__ == "__main__":
    main()
