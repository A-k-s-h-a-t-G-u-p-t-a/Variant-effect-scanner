import numpy as np
import torch
from tqdm import tqdm

TOKEN_OFFSET = 1          # verified by Gate 1 (scripts/verify.py): <cls> at index 0, <eos> at end
AAS = list("ACDEFGHIKLMNPQRSTVWY")


def _aa_ids(tok) -> list[int]:
    return [tok.convert_tokens_to_ids(aa) for aa in AAS]


@torch.no_grad()
def masked_logprobs(tok, model, sequence: str, position: int) -> dict:
    """position is 1-based. Returns {amino_acid: log_prob} for all 20."""
    ids = tok(sequence, return_tensors="pt")["input_ids"]
    idx = position - 1 + TOKEN_OFFSET

    wt = sequence[position - 1]
    assert tok.convert_ids_to_tokens(ids[0, idx].item()) == wt, \
        f"position mismatch at {position}: expected {wt}"

    ids[0, idx] = tok.mask_token_id
    logits = model(input_ids=ids).logits[0, idx]
    logp = torch.log_softmax(logits.float(), dim=-1)

    return {aa: logp[tok.convert_tokens_to_ids(aa)].item() for aa in AAS}


def score_mutation(tok, model, sequence: str, position: int, mut_aa: str) -> float:
    """More negative = more damaging."""
    wt_aa = sequence[position - 1]
    lp = masked_logprobs(tok, model, sequence, position)
    return lp[mut_aa] - lp[wt_aa]


@torch.no_grad()
def scan_logprobs(tok, model, sequence: str, batch_size: int = 8, progress=None) -> np.ndarray:
    """(L, 20) masked log-probs over the full vocab, restricted to AAS columns.

    Masks one position per row and runs `batch_size` masked copies per forward pass.
    `progress`, if given, is called with the fraction done (0..1) after each batch.
    """
    L = len(sequence)
    base = tok(sequence, return_tensors="pt")["input_ids"][0]
    aa_ids = torch.tensor(_aa_ids(tok))

    toks = tok.convert_ids_to_tokens(base)
    for p in range(1, L + 1):
        assert toks[p - 1 + TOKEN_OFFSET] == sequence[p - 1], f"position mismatch at {p}"

    out = np.zeros((L, 20), dtype=np.float32)
    positions = list(range(1, L + 1))
    for start in tqdm(range(0, L, batch_size), desc="scan"):
        batch_pos = positions[start:start + batch_size]
        idxs = torch.tensor([p - 1 + TOKEN_OFFSET for p in batch_pos])
        ids = base.repeat(len(batch_pos), 1)
        ids[torch.arange(len(batch_pos)), idxs] = tok.mask_token_id
        logits = model(input_ids=ids).logits[torch.arange(len(batch_pos)), idxs]
        logp = torch.log_softmax(logits.float(), dim=-1)
        out[start:start + len(batch_pos)] = logp[:, aa_ids].numpy()
        if progress:
            progress((start + len(batch_pos)) / L)
    return out


def logratio_from_logprobs(logprobs: np.ndarray, sequence: str) -> np.ndarray:
    """(L, 20) log p(mut) - log p(wt). WT cells are exactly 0."""
    wt_cols = [AAS.index(a) for a in sequence]
    wt_lp = logprobs[np.arange(len(sequence)), wt_cols][:, None]
    return logprobs - wt_lp


def entropy_from_logprobs(logprobs: np.ndarray) -> np.ndarray:
    """Per-position entropy (nats) of the model distribution renormalised over the 20 AAs."""
    prob = np.exp(logprobs)
    prob /= prob.sum(axis=1, keepdims=True)
    return -(prob * np.log(prob + 1e-10)).sum(axis=1)


def scan_protein(tok, model, sequence: str, batch_size: int = 8) -> np.ndarray:
    """Returns (L, 20) matrix of log-ratio scores. Rows = positions, cols = AAS."""
    return logratio_from_logprobs(scan_logprobs(tok, model, sequence, batch_size), sequence)


def position_entropy(tok, model, sequence: str, batch_size: int = 8) -> np.ndarray:
    return entropy_from_logprobs(scan_logprobs(tok, model, sequence, batch_size))
