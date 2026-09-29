"""Artifact identities and deterministic execution helpers."""

from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.distributed as dist


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_tensor_file(path):
    # Only load checkpoints from a trusted source: resumable bundles include RNG objects.
    return torch.load(path, map_location="cpu", weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


@contextlib.contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        set_rng(state)


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def autocast(device, enabled=True):
    return torch.autocast(
        device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=enabled
    )


def average_gradients(module):
    if module is None:
        return
    # All ranks participate even when a local Bernoulli branch leaves a head unused.
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if dist.is_initialized():
            dist.all_reduce(p.grad)
            p.grad.div_(dist.get_world_size())


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def append_json(value, path):
    with open(path, "a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()


def artifact_identity(path):
    """Content identity for a supplied model/embedding artifact, never just its path."""
    path = Path(path)
    if path.is_file():
        return dict(sha256=sha256(path))
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        {*path.glob("*.safetensors"), *path.glob("*.json"), *path.glob("*.jinja")}
    )
    if not files:
        raise ValueError("artifact directory contains no recognized model files")
    inventory = {p.name: sha256(p) for p in files}
    return dict(inventory_sha256=digest(inventory), files=inventory)


def runtime_identity():
    """Software/source provenance; recorded without making Git state a runtime gate."""
    import importlib.metadata
    import platform

    packages = {}
    for name in (
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "muon-optimizer",
        "math-verify",
        "evalplus",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    root = Path(__file__).parent
    sources = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}
    return dict(
        python=platform.python_version(),
        packages=packages,
        source_sha256=digest(sources),
        source_files=sources,
    )
