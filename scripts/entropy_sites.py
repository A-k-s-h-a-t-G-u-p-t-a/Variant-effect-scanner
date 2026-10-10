"""Phase 3: does ESMC's per-position confidence predict experimentally annotated functional sites?

Positives: residues in UniProt Active site / Binding site features with experimental evidence
(ECO:0000269 literature, ECO:0007744 structure), features spanning <= 5 residues.
Excluded: residues annotated only by similarity/prediction (neither positive nor negative).
"""
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS, cached_logprobs
from src.model import load_model
from src.score import AAS, entropy_from_logprobs, logratio_from_logprobs

N_PROTEINS = 25
SEED = 0
EXP_CODES = {"ECO:0000269", "ECO:0007744"}
SITE_TYPES = {"Active site", "Binding site"}
MAX_SPAN = 5
QUERY = "reviewed:true AND organism_id:9606 AND ft_act_site_exp:* AND length:[100 TO 350]"


def fetch_proteins() -> list[dict]:
    cache = DATA / "annotated_proteins.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": QUERY, "fields": "accession,id,sequence,ft_act_site,ft_binding",
         "format": "json", "size": 500})
    with urllib.request.urlopen(url) as r:
        entries = json.load(r)["results"]
    entries.sort(key=lambda e: e["primaryAccession"])
    random.Random(SEED).shuffle(entries)

    proteins = []
    for e in entries:
        seq = e["sequence"]["value"]
        if set(seq) - set(AAS):
            continue  # skip non-standard residues (U, X)
        pos_exp, pos_other = set(), set()
        for f in e.get("features", []):
            if f["type"] not in SITE_TYPES:
                continue
            s, t = f["location"]["start"]["value"], f["location"]["end"]["value"]
            if s is None or t is None or t - s + 1 > MAX_SPAN:
                continue
            codes = {ev.get("evidenceCode") for ev in f.get("evidences", [])}
            (pos_exp if codes & EXP_CODES else pos_other).update(range(s, t + 1))
        if pos_exp:
            proteins.append({"accession": e["primaryAccession"], "name": e["uniProtkbId"],
                             "sequence": seq, "sites": sorted(pos_exp),
                             "excluded": sorted(pos_other - pos_exp)})
        if len(proteins) == N_PROTEINS:
            break
    cache.write_text(json.dumps(proteins, indent=1))
    return proteins


def main():
    proteins = fetch_proteins()
    print(f"{len(proteins)} proteins, {sum(len(p['sites']) for p in proteins)} experimental site residues")

    tok = model = None
    rows, per_protein = [], []
    for p in proteins:
        acc, seq = p["accession"], p["sequence"]
        if not (RESULTS / "logprobs" / f"{acc}.npy").exists() and model is None:
            tok, model = load_model()
        print(f"{p['name']} ({len(seq)} aa, {len(p['sites'])} sites)")
        lp = cached_logprobs(tok, model, acc, seq)
        ent = entropy_from_logprobs(lp)
        lr = logratio_from_logprobs(lp, seq)
        mean_lr = (lr.sum(axis=1)) / 19  # mean over the 19 substitutions (WT cell is 0)

        excluded = set(p["excluded"])
        sites = set(p["sites"])
        keep = [i for i in range(1, len(seq) + 1) if i not in excluded]
        label = np.array([i in sites for i in keep])
        e, m = ent[np.array(keep) - 1], mean_lr[np.array(keep) - 1]
        per_protein.append({"accession": acc, "name": p["name"], "len": len(seq),
                            "n_sites": int(label.sum()),
                            "auc_entropy": roc_auc_score(label, -e),
                            "auc_mean_logratio": roc_auc_score(label, -m)})
        rows.append(pd.DataFrame({"accession": acc, "pos": keep, "site": label,
                                  "entropy": e, "mean_logratio": m}))

    df = pd.concat(rows)
    pp = pd.DataFrame(per_protein)
    df.to_csv(RESULTS / "entropy_sites_residues.csv", index=False)
    pp.to_csv(RESULTS / "entropy_sites_per_protein.csv", index=False)

    print("\n" + pp.to_string(index=False, float_format="%.3f"))
    pooled_e = roc_auc_score(df["site"], -df["entropy"])
    pooled_m = roc_auc_score(df["site"], -df["mean_logratio"])
    print(f"\npooled AUC  (-entropy):        {pooled_e:.3f}")
    print(f"pooled AUC  (-mean log-ratio): {pooled_m:.3f}")
    print(f"per-protein AUC (-entropy):    mean {pp.auc_entropy.mean():.3f}, median {pp.auc_entropy.median():.3f}")
    print(f"site residues: mean entropy {df[df.site].entropy.mean():.3f} vs others {df[~df.site].entropy.mean():.3f} nats")

    # Figure: entropy along the protein with the most sites, annotated sites shaded
    best = pp.sort_values("n_sites", ascending=False).iloc[0]
    sub = df[df.accession == best.accession]
    p = next(x for x in proteins if x["accession"] == best.accession)
    full_ent = entropy_from_logprobs(cached_logprobs(None, None, best.accession, p["sequence"]))
    fig, ax = plt.subplots(figsize=(14, 3.5))
    for s in p["sites"]:
        ax.axvspan(s - 0.5, s + 0.5, color="#f4a261", alpha=0.5, lw=0)
    ax.plot(np.arange(1, len(full_ent) + 1), full_ent, color="#264653", lw=1)
    ax.set_xlabel("residue position")
    ax.set_ylabel("entropy (nats)")
    ax.set_title(f"{best['name']}: ESMC per-position entropy, experimentally annotated "
                 f"active/binding sites shaded (AUC {best.auc_entropy:.2f})")
    ax.set_xlim(0.5, len(full_ent) + 0.5)
    plt.tight_layout()
    out = RESULTS / f"entropy_{best.accession}.png"
    plt.savefig(out, dpi=150)
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
