"""The 35 pre-scanned proteins, shared by the app and the similarity-graph build script."""
import json

import pandas as pd

from src.evaluate import DATA, RESULTS

BENCHMARK_NAMES = {
    "SUMO1_HUMAN_Weile_2017": "SUMO1, human",
    "CALM1_HUMAN_Weile_2017": "Calmodulin 1, human",
    "UBC9_HUMAN_Weile_2017": "UBC9, human",
    "DYR_ECOLI_Thompson_2019": "Dihydrofolate reductase, E. coli",
    "RASH_HUMAN_Bandaru_2017": "HRas, human",
    "GFP_AEQVI_Sarkisyan_2016": "Green fluorescent protein, jellyfish",
    "TPMT_HUMAN_Matreyek_2018": "Thiopurine methyltransferase, human",
    "BLAT_ECOLX_Stiffler_2015": "Beta-lactamase, E. coli",
    "CBX4_HUMAN_Tsuboyama_2023_2K28": "CBX4 domain, human",
}

GROUPS = {"Case study": "Case study (haemoglobin)", "Benchmark": "ProteinGym benchmark proteins",
          "Enzyme": "Enzymes with experimental functional sites"}


def read_fasta(path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">"))


def list_proteins() -> list[dict]:
    """Every preset protein that has a saved scan.

    `key` names the saved scan; `uniprot_id` is the UniProt entry name (for example HBB_HUMAN),
    used to resolve the accession and the AlphaFold structure.
    """
    proteins = [{"label": "Haemoglobin beta (P68871): sickle cell example", "key": "P68871",
                 "short": "HBB", "sequence": read_fasta(DATA / "P68871.fasta"),
                 "group": "Case study", "uniprot_id": "HBB_HUMAN"}]

    ref = pd.read_csv(DATA / "DMS_substitutions.csv").set_index("DMS_id")
    for dms_id, name in BENCHMARK_NAMES.items():
        proteins.append({"label": f"Benchmark: {name} ({dms_id})", "key": dms_id,
                         "short": ref.loc[dms_id, "UniProt_ID"].split("_")[0],
                         "sequence": ref.loc[dms_id, "target_seq"], "group": "Benchmark",
                         "dms_id": dms_id, "uniprot_id": ref.loc[dms_id, "UniProt_ID"]})

    auc = pd.read_csv(RESULTS / "entropy_sites_per_protein.csv").set_index("accession")
    for p in json.loads((DATA / "annotated_proteins.json").read_text()):
        proteins.append({"label": f"Enzyme: {p['name']} ({p['accession']})", "key": p["accession"],
                         "short": p["name"].split("_")[0], "sequence": p["sequence"],
                         "group": "Enzyme", "sites": p["sites"], "uniprot_id": p["name"],
                         "site_auc": float(auc.loc[p["accession"], "auc_entropy"])})
    return [p for p in proteins if (RESULTS / "logprobs" / f"{p['key']}.npy").exists()]
