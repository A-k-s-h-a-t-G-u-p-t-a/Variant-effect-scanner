"""Task 2.2: masked-marginal scans of the annotated proteins with another PLM (resumable).

Scans are saved to results/logprobs_<tag>/<accession>.npy (the ESMC-300M cache is untouched).
The token offset is verified for the model before any scan: the tokenizer must put exactly one
special token before residue 1 and one after the last residue, and the 20 amino-acid tokens
must all exist.

Usage: python scripts/scan_model.py --model facebook/esm2_t33_650M_UR50D --tag esm2_650m [--device auto]
Known tags: esm2_650m = facebook/esm2_t33_650M_UR50D, esmc_600m = biohub/ESMC-600M-hf
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import DATA, RESULTS, cached_logprobs
from src.model import load_model
from src.score import AAS, TOKEN_OFFSET

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


def verify_offset(tok):
    ids = tok(SEQ, return_tensors="pt")["input_ids"][0]
    toks = tok.convert_ids_to_tokens(ids)
    print("first 3 tokens:", toks[:3], "last 2:", toks[-2:], "mask:", tok.mask_token)
    assert len(toks) == len(SEQ) + TOKEN_OFFSET + 1, "expected one special token at each end"
    for p, aa in enumerate(SEQ, 1):
        assert toks[p - 1 + TOKEN_OFFSET] == aa, f"token offset wrong at {p} for this tokenizer"
    assert all(tok.convert_tokens_to_ids(a) != tok.unk_token_id for a in AAS), "missing AA token"
    assert tok.mask_token_id is not None
    print("token offset verified for this model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    subdir = f"logprobs_{args.tag}"
    proteins = json.loads((DATA / "annotated_proteins_all.json").read_text())
    todo = [p for p in proteins if not (RESULTS / subdir / f"{p['accession']}.npy").exists()]
    print(f"{args.model}: {len(proteins)} proteins, {len(proteins) - len(todo)} cached, {len(todo)} to scan")

    tok, model = load_model(args.model, device=args.device)
    verify_offset(tok)
    (RESULTS / subdir).mkdir(parents=True, exist_ok=True)
    (RESULTS / subdir / "meta.json").write_text(json.dumps({
        "model": args.model, "device": str(next(model.parameters()).device),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "protein_list": "annotated_proteins_all.json"}, indent=1))

    for i, p in enumerate(todo[:args.limit], 1):
        print(f"[{i}/{len(todo[:args.limit])}] {p['name']} ({len(p['sequence'])} aa)", flush=True)
        cached_logprobs(tok, model, p["accession"], p["sequence"], subdir=subdir)


if __name__ == "__main__":
    main()
