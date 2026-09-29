"""Read explicit recipes; paths are supplied by the caller, not embedded here."""

from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    required = {
        "task",
        "teacher",
        "representation",
        "model",
        "prompt",
        "training",
        "stages",
        "inference",
        "nft",
        "data",
    }
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"missing recipe keys: {sorted(missing)}")
    if sum(cfg["representation"]["dimensions"]) != cfg["model"]["text_encoder_dim"]:
        raise ValueError("representation dimensions must sum to latent width")
    if len(cfg["representation"]["layers"]) != len(cfg["representation"]["dimensions"]):
        raise ValueError("layer/dimension mismatch")
    if cfg["training"]["effective_batch"] < 1:
        raise ValueError("invalid effective batch")
    return cfg
