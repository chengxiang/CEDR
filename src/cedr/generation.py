"""Euler ODE sampling with feature-clock derivatives and native terminal decoding."""

from __future__ import annotations
from pathlib import Path
import json
import math
import time
import torch
from .common import (
    runtime_identity,
    autocast,
    atomic_save,
    atomic_json,
    load_tensor_file,
    sha256,
    seed_all,
    distributed_setup,
    artifact_identity,
    digest,
)
from .data import Rows, collate
from .models import build_models, encode_prompt
from .state import selected_state
from .teacher import load_embedding
from .objectives import restore


def grid(steps, mean=-1.5, std=0.8, device="cpu"):
    if steps < 1:
        raise ValueError("steps must be positive")
    p = torch.arange(1, steps, device=device, dtype=torch.float32) / steps
    q = torch.sigmoid(
        float(mean) + float(std) * (math.sqrt(2.0) * torch.erfinv(2 * p - 1))
    )
    return torch.cat([q.new_zeros(1), q, q.new_ones(1)])


def local_velocity(pred, z, local_times, dimensions, eps):
    pieces = []
    start = 0
    for t, d in zip(local_times, dimensions, strict=True):
        pieces.append(
            (pred[..., start : start + d] - z[..., start : start + d])
            / (1 - t).clamp_min(eps)[:, None, None]
        )
        start += d
    return torch.cat(pieces, -1)


@torch.no_grad()
def sample(
    model,
    condition,
    mask,
    cfg,
    *,
    generator,
    steps=None,
    powers=None,
    sccfg=None,
    cfg_scale=1.0,
    bf16=False,
    initial_noise=None,
):
    infer = cfg["inference"]
    steps = steps or infer["steps"]
    powers = powers or infer["clock_powers"]
    sccfg = infer["sccfg"] if sccfg is None else sccfg
    dims = cfg["representation"]["dimensions"]
    if len(powers) != len(dims) or any(p < 1 for p in powers):
        raise ValueError("one clock exponent >=1 is required for each feature group")
    if sum(dims) != condition.shape[-1]:
        raise ValueError("clock/latent dimensions differ")
    z = (
        torch.randn(
            condition.shape,
            device=generator.device,
            generator=generator,
            dtype=torch.float32,
        ).to(condition.device)
        * cfg["training"]["noise_scale"]
        if initial_noise is None
        else initial_noise.clone()
    ).float()
    z = restore(z, condition, mask)
    sc = restore(torch.zeros_like(z), condition, mask)
    times = grid(steps, infer["grid_mean"], infer["grid_std"], z.device)
    scale = torch.full((len(z),), float(sccfg), device=z.device)
    calls = 0
    with autocast(z.device, bf16):
        for current, following in zip(times[:-1], times[1:]):
            t, tn = current.item(), following.item()
            tau = torch.full((len(z),), t, dtype=torch.float32, device=z.device)
            gammas = torch.tensor(powers, device=z.device, dtype=torch.float32).unbind()
            local = tuple(tau.pow(gamma) for gamma in gammas)
            pred = model(
                torch.cat([z, sc], -1),
                local,
                deterministic=True,
                self_cond_cfg_scale=scale,
            )[0]
            calls += 1
            if cfg_scale != 1:
                zeros = torch.zeros_like(condition)
                uncond = model(
                    torch.cat([restore(z, zeros, mask), restore(sc, zeros, mask)], -1),
                    local,
                    deterministic=True,
                    self_cond_cfg_scale=scale,
                )[0]
                calls += 1
                pred = uncond + cfg_scale * (pred - uncond)
            pred = restore(pred, condition, mask)
            local_v = local_velocity(pred, z, local, dims, cfg["training"]["t_eps"])
            chunks = local_v.split(dims, -1)
            v = torch.cat(
                [
                    v * (power * tau.pow(power - 1.0))[:, None, None]
                    for v, power in zip(chunks, gammas)
                ],
                -1,
            )
            z = restore(z + (tn - t) * v, condition, mask)
            sc = pred
    return z, dict(denoiser_calls=calls, steps=steps, terminal_latent="euler_z")


@torch.no_grad()
def decode(model, z, sccfg, *, chunk_tokens=256, bf16=False):
    one = torch.ones(len(z), device=z.device)
    with autocast(z.device, bf16):
        _, features = model(
            torch.cat([z, torch.zeros_like(z)], -1),
            one,
            deterministic=True,
            self_cond_cfg_scale=one * sccfg,
            decoder_step_active=True,
            decoder_return_features=True,
        )
    flat = features.flatten(0, 1)
    ids = []
    with torch.autocast(z.device.type, enabled=False):
        for lo in range(0, len(flat), chunk_tokens):
            ids.append(
                (
                    flat[lo : lo + chunk_tokens].float() @ model.unembed_kernel.float()
                    + model.unembed_bias.float()
                ).argmax(-1)
            )
    return torch.cat(ids).reshape(z.shape[:2])


def run(args, cfg):
    from transformers import AutoTokenizer

    rank, world, device = distributed_setup()
    if world != 1:
        raise ValueError(
            "generation uses one process; shard input explicitly for independent GPU jobs"
        )
    package = None
    if Path(args.checkpoint).is_dir():
        from .packages import load_package

        ck, package = load_package(args.checkpoint)
        cfg = {
            **cfg,
            **{
                k: ck["config"][k]
                for k in ("task", "teacher", "representation", "data", "training")
            },
        }
        if args.prompt_source != "learned":
            raise ValueError(
                "inference packages use their matched learned prompt encoder"
            )
        args.embedding = args.embedding or str(
            Path(args.checkpoint) / package["embedding"]
        )
        args.tokenizer = args.tokenizer or str(
            Path(args.checkpoint) / package["tokenizer"]
        )
        for component in ("elf", "prompt"):
            requested = getattr(args, component + "_selector")
            if requested and requested.lstrip("0") != package["selection"][
                component
            ].lstrip("0"):
                raise ValueError(
                    f"package contains only {component} EMA {package['selection'][component]}"
                )
    else:
        ck = load_tensor_file(args.checkpoint)
    # Architecture comes from the checkpoint, evaluation settings from the recipe.
    cfg = {**cfg, "model": ck["config"]["model"], "prompt": ck["config"]["prompt"]}
    is_nft = ck.get("stage", ck.get("contract", {}).get("stage")) == "nft"
    elf_selector = (
        package["selection"]["elf"]
        if package
        else args.elf_selector
        or (
            cfg["nft"]["elf_eval_selector"]
            if is_nft
            else cfg["inference"]["elf_selector"]
        )
    )
    prompt_selector = (
        package["selection"]["prompt"]
        if package
        else args.prompt_selector
        or (
            cfg["nft"]["prompt_eval_selector"]
            if is_nft
            else cfg["inference"]["prompt_selector"]
        )
    )
    embedding = (
        load_embedding(args.embedding) if args.prompt_source == "learned" else None
    )
    seed_all(args.seed)
    torch.set_float32_matmul_precision(cfg["inference"].get("matmul_precision", "high"))
    elf, prompt = build_models(
        cfg,
        device,
        embedding,
        stage="joint" if args.prompt_source == "learned" else "flow",
    )
    elf.load_state_dict(
        selected_state(ck, "elf", "raw" if package else elf_selector), strict=True
    )
    elf.eval()
    if prompt:
        prompt.load_state_dict(
            selected_state(ck, "prompt", "raw" if package else prompt_selector),
            strict=True,
        )
        prompt.eval()
    dataset = Rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or cfg["teacher"]["model_id"],
        revision=cfg["teacher"]["revision"],
    )
    provider = None
    if args.prompt_source == "teacher":
        from .train import feature_provider

        provider = feature_provider(args, cfg, dataset, device)
    settings = dict(
        recipe_sha256=digest(cfg),
        feature_provider=None if provider is None else provider.identity,
        checkpoint_sha256=sha256(
            Path(args.checkpoint) / "manifest.json" if package else args.checkpoint
        ),
        data_sha256=dataset.identity,
        seed=args.seed,
        elf_selector=elf_selector,
        prompt_selector=prompt_selector,
        prompt_source=args.prompt_source,
        steps=args.steps or cfg["inference"]["steps"],
        powers=args.powers or cfg["inference"]["clock_powers"],
        sccfg=args.sccfg if args.sccfg is not None else cfg["inference"]["sccfg"],
        cfg=args.cfg,
        batch_size=args.batch_size,
        embedding=None if not args.embedding else artifact_identity(args.embedding),
        arithmetic=cfg["inference"].get("arithmetic", "fp32"),
        noise_generator="cpu",
        decoder="argmax",
        max_response_tokens=cfg["inference"].get("max_response_tokens"),
        terminal_latent="euler_z",
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    predictions = []
    timings = []
    progress = out / "progress.pt"
    if progress.exists():
        previous = load_tensor_file(progress)
        if previous["settings"] != settings:
            raise ValueError("generation continuation settings changed")
        predictions = previous["predictions"]
        timings = previous["timings"]
        gen.set_state(previous["rng"].cpu())
    terminal = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for lo in range(len(predictions), len(dataset), args.batch_size):
        rows = [dataset[i] for i in range(lo, min(lo + args.batch_size, len(dataset)))]
        # Evaluation always uses the full canvas and never reads answer lengths.
        for row in rows:
            row["input_ids"] = row["input_ids"][: row["prompt_length"]]
            row["content_end"] = row["prompt_length"]
        batch = collate(rows, cfg["data"]["max_tokens"], tokenizer.pad_token_id, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        with torch.no_grad():
            with autocast(
                device, cfg["inference"].get("prompt_arithmetic", "fp32") == "bf16"
            ):
                condition = (
                    encode_prompt(
                        prompt,
                        batch,
                        True,
                        pad_to_multiple=cfg["inference"].get(
                            "prompt_padding_multiple", 1
                        ),
                        pad_id=tokenizer.pad_token_id,
                    )
                    if prompt
                    else provider(batch).float()
                )
        z, counts = sample(
            elf,
            condition,
            batch["prompt"],
            cfg,
            generator=gen,
            steps=settings["steps"],
            powers=settings["powers"],
            sccfg=settings["sccfg"],
            cfg_scale=args.cfg,
            bf16=settings["arithmetic"] == "bf16",
        )
        ids = decode(elf, z, settings["sccfg"], bf16=settings["arithmetic"] == "bf16")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        for row, tokens in zip(rows, ids.cpu().tolist()):
            tokens = tokens[row["prompt_length"] :]
            if settings["max_response_tokens"] is not None:
                tokens = tokens[: settings["max_response_tokens"]]
            if terminal in tokens:
                tokens = tokens[: tokens.index(terminal)]
            predictions.append(
                dict(
                    id=row["id"],
                    seed=args.seed,
                    prediction=tokenizer.decode(tokens, skip_special_tokens=True),
                    token_ids=tokens,
                )
            )
        timings.append(
            dict(
                first=lo,
                rows=len(rows),
                seconds=elapsed,
                decoder_calls=1,
                prompt_calls=int(prompt is not None),
                **counts,
            )
        )
        atomic_save(
            dict(
                settings=settings,
                predictions=predictions,
                timings=timings,
                rng=gen.get_state(),
            ),
            progress,
        )
    path = out / "predictions.jsonl"
    tmp = out / "predictions.tmp"
    tmp.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    tmp.replace(path)
    atomic_json(
        dict(
            status="completed",
            runtime=runtime_identity(),
            settings=settings,
            coverage=len(predictions),
            prediction_sha256=sha256(path),
            timings=timings,
        ),
        out / "generation.json",
    )
