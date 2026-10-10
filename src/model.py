import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "biohub/ESMC-300M-hf"


def load_model(name: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name, dtype=torch.float32).eval()
    torch.set_num_threads(4)
    return tok, model
