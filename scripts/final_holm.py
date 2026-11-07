"""A4: one Holm-corrected family over every pre-registered contrast (Phase A configs, B1 strata, C2 label
types), as written in results/analysis_plan.md. C1 (confirmatory) and C3 (consistency check) are
reported separately and are not part of the family. Writes results/holm_family.csv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cvtools import holm
from src.evaluate import RESULTS


def main():
    a = pd.read_csv(RESULTS / "phase_a_contrasts.csv")
    a = pd.DataFrame({"group": "A", "test": a.config + " | " + a.metric, "diff": a["diff"],
                      "ci_lo": a.ci_lo, "ci_hi": a.ci_hi, "p_raw": a.p_boot})
    b = pd.read_csv(RESULTS / "b1_strata.csv")
    b = b[b.proxy != "annotation_score"]  # constant (5) for all 126 proteins: terciles would split ties arbitrarily
    b = pd.DataFrame({"group": "B1", "test": b.proxy + " " + b.tercile + " | " + b.metric, "diff": b["diff"],
                      "ci_lo": b.ci_lo, "ci_hi": b.ci_hi, "p_raw": b.p_boot})
    c = pd.read_csv(RESULTS / "c2_labels.csv")
    c = pd.DataFrame({"group": "C2", "test": c.label_type + " | " + c.metric, "diff": c["diff"],
                      "ci_lo": c.ci_lo, "ci_hi": c.ci_hi, "p_raw": c.p_boot})
    fam = pd.concat([a, b, c], ignore_index=True)
    fam["p_holm"] = holm(fam.p_raw.to_numpy())
    fam.to_csv(RESULTS / "holm_family.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(fam)} tests in the family; raw p < 0.05: {(fam.p_raw < 0.05).sum()}, "
          f"Holm p < 0.05: {(fam.p_holm < 0.05).sum()}")
    print(fam.sort_values("p_raw").head(12).to_string(index=False, float_format="%.4f"))
    print("\npositive point estimates:", (fam["diff"] > 0).sum(), "of", len(fam))


if __name__ == "__main__":
    main()
