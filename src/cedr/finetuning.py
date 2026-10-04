"""Initialize joint fine-tuning from a selected model package."""

from copy import deepcopy
from pathlib import Path

from .common import load_tensor_file, sha256
from .features import Representation
from .packages import load_package, safe_path


def effective_config(cfg, source):
    """Keep optimization choices while taking architecture from the weights."""
    cfg = deepcopy(cfg)
    if cfg["task"] != source["task"]:
        raise ValueError("training recipe and model package have different tasks")
    for key in ("model_id", "revision"):
        if cfg["teacher"][key] != source["teacher"][key]:
            raise ValueError("training recipe and model package use different teachers")
    for key in ("layers", "dimensions"):
        if cfg["representation"][key] != source["representation"][key]:
            raise ValueError("training recipe and model package representations differ")
    for key in ("model", "prompt"):
        cfg[key] = deepcopy(source[key])
    if cfg["data"]["max_tokens"] > min(
        cfg["model"]["max_length"], cfg["prompt"]["max_length"]
    ):
        raise ValueError("training canvas exceeds the model package capacity")
    return cfg


def resolve(args, cfg):
    """Return effective config, optional weights/resume state, and initialization ID."""
    directory = getattr(args, "init_weights", None)
    if directory:
        if args.stage != "joint":
            raise ValueError("--init-weights is supported for joint training")
        if any((args.init, args.prompt_init, args.resume)):
            raise ValueError("--init-weights cannot be combined with --init, --prompt-init, or --resume")
        if args.elf_selector or args.prompt_selector:
            raise ValueError("--init-weights uses the package's already-selected pair")
        root = Path(directory)
        bundle, manifest = load_package(root)
        if not manifest.get("representation"):
            raise ValueError("model package has no representation; download the updated package")
        cfg = effective_config(cfg, bundle["config"])
        for argument, key in (("embedding", "embedding"), ("representation", "representation")):
            expected = manifest["files"][manifest[key]]["sha256"]
            path = getattr(args, argument) or safe_path(root, manifest[key])
            if sha256(path) != expected:
                raise ValueError(f"{argument} does not match the model package")
            setattr(args, argument, str(Path(path).resolve()))
        rep = Representation(args.representation)
        if (
            rep.layers != cfg["representation"]["layers"]
            or rep.dimensions != cfg["representation"]["dimensions"]
            or rep.payload.get("teacher") != {
                key: cfg["teacher"][key] for key in ("model_id", "revision")
            }
            or manifest["model_id"] not in rep.payload.get("compatible_models", [])
        ):
            raise ValueError("representation metadata does not match the model package")
        for mean, encoder in rep.blocks:
            if mean.shape != (cfg["prompt"]["external_embedding_dim"],):
                raise ValueError("representation hidden width differs from the prompt embedding")
        args.tokenizer = str(safe_path(root, manifest["tokenizer"]))
        identity = dict(
            model_id=manifest["model_id"],
            manifest_sha256=sha256(root / "manifest.json"),
            selection=manifest["selection"],
            representation_sha256=rep.identity,
            embedding_sha256=manifest["files"][manifest["embedding"]]["sha256"],
        )
        return cfg, bundle, None, identity
    if args.resume:
        parent = load_tensor_file(args.resume)
        identity = parent["contract"].get("weights_initialization")
        if identity:
            if args.stage != "joint" or args.init or args.prompt_init:
                raise ValueError("fine-tuning resume requires joint training without initialization arguments")
            cfg = effective_config(cfg, parent["config"])
            assets = parent["extra"]["finetuning_assets"]
            for name in ("embedding", "representation", "tokenizer"):
                if not getattr(args, name, None):
                    setattr(args, name, assets[name])
            for name in ("embedding", "representation"):
                if sha256(getattr(args, name)) != identity[name + "_sha256"]:
                    raise ValueError(f"{name} differs from the fine-tuning checkpoint")
        return cfg, None, parent, identity
    return cfg, None, None, None


def validate_data(dataset, cfg, tokenizer_path):
    """Check the prepared-data contract before reading training features."""
    manifest = dataset.manifest
    if manifest.get("inference", False):
        raise ValueError("joint fine-tuning requires solutions, not inference-only data")
    if manifest.get("task") != cfg["task"]:
        raise ValueError("prepared data and training recipe have different tasks")
    teacher = manifest.get("teacher", {})
    if any(teacher.get(k) != cfg["teacher"][k] for k in ("model_id", "revision")):
        raise ValueError("prepared data uses a different teacher/tokenizer revision")
    if manifest.get("data", {}).get("max_tokens", 0) > cfg["data"]["max_tokens"]:
        raise ValueError("prepared data exceeds the training canvas")
    if manifest.get("tokenizer_sha256"):
        from transformers import AutoTokenizer
        from .data import tokenizer_fingerprint

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_sha256"]:
            raise ValueError("prepared data tokenizer differs from the model package")
