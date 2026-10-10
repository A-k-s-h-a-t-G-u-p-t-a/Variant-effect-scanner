"""Gate 4: zero-shot Spearman on a set of small ProteinGym assays, vs published baselines."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS, evaluate_assay
from src.model import load_model

ASSAYS = [
    "SUMO1_HUMAN_Weile_2017",
    "CALM1_HUMAN_Weile_2017",
    "UBC9_HUMAN_Weile_2017",
    "DYR_ECOLI_Thompson_2019",
    "RASH_HUMAN_Bandaru_2017",
    "GFP_AEQVI_Sarkisyan_2016",
    "TPMT_HUMAN_Matreyek_2018",
    "BLAT_ECOLX_Stiffler_2015",
    "CBX4_HUMAN_Tsuboyama_2023_2K28",
]
BASELINES = ["ESM-C (300M)", "ESM2 (650M)", "ESM-1v (single)"]


def main():
    ref = pd.read_csv(DATA / "DMS_substitutions.csv").set_index("DMS_id")
    pub = pd.read_csv(DATA / "pg_spearman.csv")
    pub["DMS_id"] = pub["DMS ID"].str.removesuffix(".csv")
    pub = pub.set_index("DMS_id")

    tok, model = load_model()
    rows = []
    for dms_id in ASSAYS:
        print(f"\n{dms_id} ({ref.loc[dms_id, 'seq_len']} aa)")
        r = evaluate_assay(tok, model, dms_id, ref.loc[dms_id, "target_seq"])
        r["selection_type"] = ref.loc[dms_id, "coarse_selection_type"]
        for b in BASELINES:
            r[f"pub {b}"] = pub.loc[dms_id, b] if dms_id in pub.index else float("nan")
        print(f"  spearman = {r['spearman']:.3f}   (published ESM-C 300M: {r['pub ESM-C (300M)']:.3f})")
        rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "proteingym_spearman.csv", index=False)
    cols = ["DMS_id", "seq_len", "n_mutants", "selection_type", "spearman", "spearman_singles"] \
        + [f"pub {b}" for b in BASELINES]
    print("\n" + res[cols].to_string(index=False, float_format="%.3f"))
    print("\nmean:", res[["spearman"] + [f"pub {b}" for b in BASELINES]].mean().round(3).to_dict())
    print("GATE 4", "PASS" if res["spearman"].mean() > 0.25 else "FAIL")


if __name__ == "__main__":
    main()
