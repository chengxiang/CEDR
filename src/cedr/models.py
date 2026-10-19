"""Diffusion backbone and prompt encoder with frozen token embeddings."""

import torch
import torch.nn.functional as F
from .nn.model import CEDR
from .nn.prompt import (
    FrozenQwenEmbeddingPromptEncoder,
    FrozenQwenEmbeddingPromptEncoderConfig,
)
from .optim import muon_with_aux_adam


def build_models(cfg, device, embedding=None, stage="joint"):
    elf = None if stage == "prompt" else CEDR(**cfg["model"]).to(device)
    prompt = None
    if stage != "flow":
        if embedding is None:
            raise ValueError("a frozen teacher embedding artifact is required")
        prompt = FrozenQwenEmbeddingPromptEncoder(
            FrozenQwenEmbeddingPromptEncoderConfig(**cfg["prompt"]),
            embedding.to(device),
        ).to(device)
    return elf, prompt


def encode_prompt(prompt, batch, deterministic=False, *, pad_to_multiple=1, pad_id=0):
    span = int(batch["prompt"].sum(1).max())
    span = ((span + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    if span > batch["ids"].shape[1]:
        raise ValueError("padded prompt exceeds canvas")
    mask = batch["prompt"][:, :span]
    # Never even look up answer IDs in the student.
    ids = torch.where(
        mask, batch["ids"][:, :span], torch.full_like(batch["ids"][:, :span], pad_id)
    )
    result = prompt(ids, mask, deterministic=deterministic)
    return F.pad(result, (0, 0, 0, batch["ids"].shape[1] - span)).float()


def make_optimizers(elf, prompt, lr, prompt_lr=None):
    prompt_lr = lr if prompt_lr is None else prompt_lr
    optimizers = {}
    if elf is not None:
        optimizers["elf"] = [muon_with_aux_adam(elf, lr)]
    if prompt is not None:
        matrix = [
            (k, p)
            for k, p in prompt.named_parameters()
            if p.requires_grad and p.ndim == 2
        ]
        aux = [p for p in prompt.parameters() if p.requires_grad and p.ndim != 2]
        optimizers["prompt"] = [
            muon_with_aux_adam(prompt, prompt_lr, named_parameters=matrix),
            torch.optim.AdamW(
                aux, lr=prompt_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
            ),
        ]
    return optimizers


def promote_time_embedders(state, names):
    """Clone the synchronous time embedder into independent NFT group embedders."""
    out = {k: v for k, v in state.items() if not k.startswith("t_embedder.")}
    for k, v in state.items():
        if k.startswith("t_embedder."):
            for name in names:
                out["t_embedders." + name + "." + k[len("t_embedder.") :]] = v.clone()
    return out
