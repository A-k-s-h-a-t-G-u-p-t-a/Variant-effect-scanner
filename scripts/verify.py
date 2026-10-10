"""Gate 1: print model config and empirically verify the token offset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import load_model

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"

tok, model = load_model()

print("num_hidden_layers:", model.config.num_hidden_layers)
print("hidden_size:      ", model.config.hidden_size)
print("mask token:       ", tok.mask_token, tok.mask_token_id)

ids = tok(SEQ, return_tensors="pt")["input_ids"]
toks = tok.convert_ids_to_tokens(ids[0])

print("sequence length:", len(SEQ))
print("token count:    ", len(toks))
print("first 5 tokens: ", toks[:5])
print("last 3 tokens:  ", toks[-3:])

# find the offset empirically — do not hardcode a guess
OFFSET = next(i for i, t in enumerate(toks) if t == SEQ[0])
print("TOKEN_OFFSET =", OFFSET)

# lock it in: residue p (1-based) must sit at index p-1+OFFSET
for p in range(1, len(SEQ) + 1):
    assert toks[p - 1 + OFFSET] == SEQ[p - 1], f"offset wrong at position {p}"
assert len(toks) == len(SEQ) + OFFSET + 1, "expected exactly one special token at each end"
print("offset verified OK")
