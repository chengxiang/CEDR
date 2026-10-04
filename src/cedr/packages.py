"""Verified, selected-weight inference packages on the Hugging Face Hub."""

from __future__ import annotations
import json
import re
from pathlib import Path, PurePosixPath
from safetensors.torch import load_file
from .common import atomic_json, sha256

REPOSITORY = "xc91/CEDR"
MODELS = tuple(
    f"{family}-{stage}-nft"
    for family in ("gsm8k-b", "math-l", "oci-l")
    for stage in ("pre", "post")
)


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"invalid package path: {name}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"package path escapes root: {name}")
    return path


def verify_files(root, files):
    for name, record in files.items():
        path = safe_path(root, name)
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"package file failed verification: {name}")


def read_package(directory, verify=True):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "cedr-inference-v1":
        raise ValueError("unsupported inference package format")
    required = {
        manifest["config"],
        manifest["embedding"],
        *manifest["weights"].values(),
    }
    if manifest.get("representation"):
        required.add(manifest["representation"])
    if not required.issubset(manifest["files"]):
        raise ValueError("package is missing required file hashes")
    if set(manifest["weights"]) != {"elf", "prompt"}:
        raise ValueError("package must contain one model/prompt pair")
    if verify:
        verify_files(root, manifest["files"])
    config = json.loads(safe_path(root, manifest["config"]).read_text())
    return manifest, config


def load_package(directory):
    root = Path(directory)
    manifest, config = read_package(root)
    states = {
        key: load_file(str(safe_path(root, manifest["weights"][key])))
        for key in ("elf", "prompt")
    }
    return (
        dict(
            format="cedr-export-v1",
            config=config,
            models=states,
            emas={},
            selection=manifest["selection"],
        ),
        manifest,
    )


def download(model, output, revision=None):
    """Resolve one immutable revision, then verify every downloaded file."""
    from huggingface_hub import HfApi, hf_hub_download

    if model not in MODELS:
        raise ValueError(f"unknown model {model}; choose from {MODELS}")
    resolved = HfApi().model_info(REPOSITORY, revision=revision or "main").sha
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Hub did not return an immutable revision")
    catalog = json.loads(
        Path(hf_hub_download(REPOSITORY, "index.json", revision=resolved)).read_text()
    )
    item = catalog["models"][model]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    import shutil

    for local_name, record in item["files"].items():
        destination = safe_path(root, local_name)
        source_name = record["path"]
        safe_path(Path("/"), source_name)
        source = Path(hf_hub_download(REPOSITORY, source_name, revision=resolved))
        if (
            source.stat().st_size != record["bytes"]
            or sha256(source) != record["sha256"]
        ):
            raise ValueError(f"Hub file failed verification: {source_name}")
        if destination.exists() and sha256(destination) == record["sha256"]:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    read_package(root)
    atomic_json(
        dict(repository=REPOSITORY, revision=resolved, model=model),
        root / "download.json",
    )
    return root
