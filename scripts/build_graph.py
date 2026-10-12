"""Build the protein similarity graph for the app.

Phases: UniProt metadata -> embeddings -> AlphaFold structures -> thumbnails -> substitution
fingerprints -> sequence identity -> graphs for both views -> remote-homologue analysis.
Everything is cached; rerunning is fast.
"""
import io
import json
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS, cached_logprobs
from src.graph import (alignment_zscore, knn_edges, mds_layout, percent_identity, similarity,
                       standardise, substitution_fingerprint)
from src.presets import list_proteins
from src.score import logratio_from_logprobs
from src.structures import THUMB_DIR, fetch_structure, render_thumbnail

K = 3
ALIGN_FINDS_Z = 8.0     # shuffle z-score at or above this: sequence alignment alone reveals the relationship
NO_SIGNAL_Z = 5.0       # below this: alignment detects nothing beyond chance


def uniprot_metadata(entry_names: list[str]) -> pd.DataFrame:
    cache = DATA / "protein_metadata.csv"
    if cache.exists():
        df = pd.read_csv(cache, keep_default_na=False)
    else:
        query = " OR ".join(f"id:{n}" for n in entry_names)
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": query, "format": "tsv", "size": 500,
             "fields": "accession,id,protein_name,organism_name,length,xref_pfam,xref_interpro"})
        df = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text), sep="\t", keep_default_na=False)
        df.to_csv(cache, index=False)
    df = df[df["Organism"] != ""]   # drop retired ("demerged") entries that still carry the old name
    missing = set(entry_names) - set(df["Entry Name"])
    assert not missing, f"UniProt did not resolve: {missing}"
    assert not df["Entry Name"].duplicated().any(), "ambiguous UniProt entry names"
    return df.set_index("Entry Name")


def embeddings(proteins) -> np.ndarray:
    path = RESULTS / "embeddings.npz"
    cached = dict(np.load(path)) if path.exists() else {}
    todo = [p for p in proteins if p["key"] not in cached]
    if todo:
        from src.embed import embed_sequence
        from src.model import load_model
        tok, model = load_model()
        for p in todo:
            print(f"  embedding {p['short']} ({len(p['sequence'])} aa)")
            cached[p["key"]] = embed_sequence(tok, model, p["sequence"])
        np.savez(path, **cached)
    return np.stack([cached[p["key"]] for p in proteins])


def alignment_matrices(proteins) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise local-alignment percent identity and shuffle z-score (cached)."""
    path = RESULTS / "alignment.npz"
    keys = [p["key"] for p in proteins]
    if path.exists():
        z = np.load(path)
        if list(z["keys"]) == keys:
            return z["identity"], z["zscore"]
    n = len(proteins)
    ident, zs = np.full((n, n), 100.0), np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = proteins[i]["sequence"], proteins[j]["sequence"]
            ident[i, j] = ident[j, i] = percent_identity(a, b)
            zs[i, j] = zs[j, i] = alignment_zscore(a, b)
    np.savez(path, keys=np.array(keys), identity=ident, zscore=zs)
    return ident, zs


def split(field: str) -> set[str]:
    return {x for x in field.split(";") if x}


def main():
    proteins = list_proteins()
    n = len(proteins)
    print(f"{n} proteins")

    print("UniProt metadata")
    meta = uniprot_metadata([p["uniprot_id"] for p in proteins])

    print("Embeddings")
    X_emb = embeddings(proteins)

    print("AlphaFold structures and thumbnails")
    nodes = []
    for p in proteins:
        m = meta.loc[p["uniprot_id"]]
        acc = m["Entry"]
        info = fetch_structure(acc)
        thumb = THUMB_DIR / f"{acc}.png"
        if info and not thumb.exists():
            render_thumbnail(DATA / "structures" / f"{acc}.pdb", thumb)
        print(f"  {p['short']:8s} {acc:8s} structure: "
              + (f"{info['entry_id']} (mean pLDDT {info['plddt']:.1f})" if info else "none"))
        nodes.append({
            "key": p["key"], "short": p["short"], "label": p["label"], "group": p["group"],
            "accession": acc, "uniprot_id": p["uniprot_id"], "name": m["Protein names"].split(" (")[0],
            "organism": m["Organism"].split(" (")[0], "scanned_length": len(p["sequence"]),
            "uniprot_length": int(m["Length"]), "pfam": sorted(split(m["Pfam"])),
            "interpro": sorted(split(m["InterPro"])),
            "structure": f"data/structures/{acc}.pdb" if info else None,
            "plddt": info["plddt"] if info else None,
            "thumbnail": f"results/thumbnails/{acc}.png" if info else None,
            "dms_id": p.get("dms_id"), "site_auc": p.get("site_auc"),
        })

    print("Substitution fingerprints")
    X_sub = np.stack([substitution_fingerprint(
        logratio_from_logprobs(cached_logprobs(None, None, p["key"], p["sequence"]), p["sequence"]),
        p["sequence"]) for p in proteins])

    print("Sequence alignment: identity and shuffle z-score (595 pairs, about a minute)")
    ident, zs = alignment_matrices(proteins)

    views = {}
    for name, X in [("embedding", X_emb), ("substitution", standardise(X_sub))]:
        S = similarity(X)
        views[name] = {"similarity": S.round(5).tolist(), "edges": knn_edges(S, K),
                       "pos": mds_layout(S).round(5).tolist()}

    # ---- analysis: does embedding similarity find relationships that alignment misses?
    iu = np.triu_indices(n, 1)
    rows = []
    for i, j in zip(*iu):
        a, b = nodes[i], nodes[j]
        rows.append({"protein_a": a["short"], "protein_b": b["short"],
                     "alignment_z": zs[i, j], "identity": ident[i, j],
                     "embedding_cosine": views["embedding"]["similarity"][i][j],
                     "substitution_cosine": views["substitution"]["similarity"][i][j],
                     "shared_pfam": ";".join(sorted(set(a["pfam"]) & set(b["pfam"]))),
                     "shared_interpro": ";".join(sorted(set(a["interpro"]) & set(b["interpro"]))),
                     "name_a": a["name"], "name_b": b["name"]})
    pairs = pd.DataFrame(rows)
    for v in ("embedding", "substitution"):
        pairs[f"{v}_rank"] = pairs[f"{v}_cosine"].rank(ascending=False).astype(int)
    pairs.to_csv(RESULTS / "protein_pairs.csv", index=False)

    stats = {
        "spearman_z_embedding": spearmanr(pairs.alignment_z, pairs.embedding_cosine).correlation,
        "spearman_z_substitution": spearmanr(pairs.alignment_z, pairs.substitution_cosine).correlation,
        "spearman_embedding_substitution":
            spearmanr(pairs.embedding_cosine, pairs.substitution_cosine).correlation,
    }
    related = pairs[pairs.alignment_z >= ALIGN_FINDS_Z]
    print(f"\n{len(pairs)} pairs; alignment finds {len(related)} related pair(s) (z >= {ALIGN_FINDS_Z:.0f})")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}")
    print("\nPairs that alignment finds, and where each view ranks them (out of 595):")
    print(related[["protein_a", "protein_b", "alignment_z", "identity", "embedding_rank",
                   "substitution_rank", "shared_pfam"]].to_string(index=False, float_format="%.1f"))

    none = pairs[pairs.alignment_z < NO_SIGNAL_Z].copy()
    none["shares_interpro"] = none.shared_interpro != ""
    if none.shares_interpro.sum() >= 5:   # an area under the curve on fewer positives is meaningless
        for v in ("embedding", "substitution"):
            stats[f"auc_{v}_shared_interpro_without_alignment"] = roc_auc_score(
                none.shares_interpro, none[f"{v}_cosine"])
    top = none.sort_values("embedding_cosine", ascending=False).head(20)
    print(f"\nPairs with no detectable alignment (z < {NO_SIGNAL_Z:.0f}): {len(none)}; "
          f"{none.shares_interpro.mean():.1%} share an InterPro entry "
          f"(top 20 by embedding cosine: {top.shares_interpro.mean():.0%})")
    for v in ("embedding", "substitution"):
        key = f"auc_{v}_shared_interpro_without_alignment"
        if key in stats:
            print(f"  area under the curve, {v} cosine predicting a shared InterPro entry: {stats[key]:.3f}")
    print("\nTop no-alignment / high-embedding-similarity pairs (candidates to check by hand):")
    print(top.head(8)[["protein_a", "protein_b", "alignment_z", "embedding_cosine", "embedding_rank",
                       "shared_interpro", "name_a", "name_b"]].to_string(index=False, float_format="%.3f"))

    zs_out = np.where(np.isfinite(zs), zs, 999.0)
    out = {"k": K, "align_finds_z": ALIGN_FINDS_Z, "no_signal_z": NO_SIGNAL_Z, "nodes": nodes,
           "identity": ident.round(2).tolist(), "alignment_z": zs_out.round(2).tolist(),
           "views": views, "stats": stats}
    (RESULTS / "graph.json").write_text(json.dumps(out))
    print("\nsaved results/graph.json, results/protein_pairs.csv")


if __name__ == "__main__":
    main()
