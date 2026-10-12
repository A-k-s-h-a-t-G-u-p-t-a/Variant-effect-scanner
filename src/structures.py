"""AlphaFold structure download and alpha-carbon trace thumbnails."""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from src.evaluate import DATA, RESULTS

STRUCT_DIR = DATA / "structures"
THUMB_DIR = RESULTS / "thumbnails"


def alphafold_info(accession: str) -> dict | None:
    """Resolve the current model through the API (file versions change; never template the URL)."""
    r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=30)
    if r.status_code != 200 or not r.json():
        return None
    p = r.json()[0]
    return {"pdb_url": p.get("pdbUrl"), "entry_id": p.get("entryId"),
            "plddt": p.get("globalMetricValue"),
            "start": p.get("uniprotStart"), "end": p.get("uniprotEnd")}


def fetch_structure(accession: str) -> dict | None:
    """Download (once) the AlphaFold PDB for an accession. Returns its info, or None if unavailable."""
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = STRUCT_DIR / f"{accession}.json"
    pdb_path = STRUCT_DIR / f"{accession}.pdb"
    if meta_path.exists():
        info = json.loads(meta_path.read_text())
        return info if info and pdb_path.exists() else None
    try:
        info = alphafold_info(accession)
        time.sleep(1)  # be polite to the free public service
        if info and info["pdb_url"]:
            r = requests.get(info["pdb_url"], timeout=60)
            r.raise_for_status()
            pdb_path.write_text(r.text)
            time.sleep(1)
    except requests.RequestException as e:
        print(f"  {accession}: download failed ({e})")
        return None  # not cached, so the next run retries
    meta_path.write_text(json.dumps(info))
    return info if info and pdb_path.exists() else None


def parse_ca(pdb_path) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) alpha-carbon coordinates and per-residue pLDDT (AlphaFold stores it as the B-factor)."""
    coords, plddt = [], []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            plddt.append(float(line[60:66]))
    return np.array(coords), np.array(plddt)


def render_thumbnail(pdb_path, out_png, size=(3.2, 3.2)):
    """Alpha-carbon trace coloured from N-terminus (blue) to C-terminus (red), the structural convention."""
    xyz, _ = parse_ca(pdb_path)
    xyz = xyz - xyz.mean(axis=0)
    # rotate onto principal axes so the widest extent faces the viewer
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    xyz = xyz @ vt.T
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    t = np.linspace(0, 1, len(xyz))
    for i in range(len(xyz) - 1):
        ax.plot(*xyz[i:i + 2].T, color=plt.cm.rainbow(t[i]), linewidth=1.8)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.set_box_aspect([np.ptp(xyz[:, i]) or 1 for i in range(3)])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=90, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
