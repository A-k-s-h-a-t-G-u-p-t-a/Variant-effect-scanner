import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "biohub/ESMC-300M-hf"


def load_model(name: str = MODEL_NAME, device: str = "cpu"):
    """device: "cpu" (default), "cuda", or "auto" (cuda if available). Always float32."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name, dtype=torch.float32).eval().to(device)
    torch.set_num_threads(4)
    return tok, model
