"""Torchrun loop for flow, standalone prompt, and joint training."""

from __future__ import annotations
import json
import math
from pathlib import Path
import signal
import time
import torch
import torch.distributed as dist
from .balance import PromptMSEBalance, row_mse
from .common import (
    runtime_identity,
    distributed_setup,
    seed_all,
    preserve_rng,
    load_tensor_file,
    sha256,
    digest,
    autocast,
    average_gradients,
    append_json,
    atomic_json,
    artifact_identity,
)
from .data import Rows, Cursor, collate
from .features import CachedFeatures, LiveFeatures, Representation
from .teacher import QwenTeacher, load_embedding
from .models import build_models, encode_prompt, make_optimizers
from .objectives import mixed_loss, restore
from .state import EMA, save_training, restore_training, selected_state


def feature_provider(args, cfg, dataset, device):
    if args.features == "cache":
        if not args.cache:
            raise ValueError("--cache is required for cached training")
        return CachedFeatures(args.cache, dataset, args.representation)
    if not args.representation:
        raise ValueError("--representation is required for live training")
    representation = Representation(args.representation, device)
    teacher = QwenTeacher(
        args.teacher or cfg["teacher"]["model_id"],
        representation.layers,
        device,
        revision=cfg["teacher"]["revision"],
        chunk_rows=args.qwen_chunk or cfg["teacher"]["chunk_rows"],
    )
    return LiveFeatures(teacher, representation)


def run(args, cfg):
    if args.stage == "nft":
        from .nft import run as nft_run

        return nft_run(args, cfg)
    rank, world, device = distributed_setup()
    start = time.monotonic()
    c = cfg["training"]
    stage = args.stage
    dataset = Rows(args.data, prompts_only=stage == "prompt")
    provider = feature_provider(args, cfg, dataset, device)
    embedding = load_embedding(args.embedding) if stage != "flow" else None
    with preserve_rng():
        seed_all(c["seed"])
        elf, prompt = build_models(cfg, device, embedding, stage)
    models = {k: m for k, m in dict(elf=elf, prompt=prompt).items() if m is not None}
    optimizers = make_optimizers(elf, prompt, c["lr"])
    emas = {k: EMA(m, c["ema_decays"]) for k, m in models.items()}
    if stage == "joint" and not args.resume:
        if not args.init or not args.prompt_init:
            raise ValueError(
                "joint needs --init flow checkpoint and --prompt-init standalone checkpoint"
            )
        parent = load_tensor_file(args.init)
        elf.load_state_dict(parent["models"]["elf"], strict=True)
        emas["elf"].load_state_dict(parent["emas"]["elf"])
        if "elf" not in parent.get("optimizers", {}):
            raise ValueError(
                "joint needs mature ELF optimizer state, not an inference export"
            )
        for o, s in zip(optimizers["elf"], parent["optimizers"]["elf"], strict=True):
            o.load_state_dict(s)
        pinit = load_tensor_file(args.prompt_init)
        prompt.load_state_dict(
            selected_state(
                pinit,
                "prompt",
                args.prompt_selector or cfg["stages"]["joint_prompt_selector"],
            ),
            strict=True,
        )
        emas["prompt"] = EMA(prompt, c["ema_decays"])
    elif args.init and not args.resume:
        raise ValueError(
            "--init is a joint stage transition; use --resume for continuation"
        )
    balance = (
        PromptMSEBalance(
            decay=c["mse_balance_decay"], target_ratio=c["prompt_mse_ratio"]
        )
        if stage == "joint" and c["prompt_mse_ratio"] > 0
        else None
    )
    micro = args.micro_batch or (
        c["effective_batch"] // world if stage == "prompt" else c["micro_batch"]
    )
    global_micro = micro * world
    if c["effective_batch"] % global_micro:
        raise ValueError("effective batch must divide micro_batch × world size")
    accumulation = c["effective_batch"] // global_micro
    cursor = Cursor(
        len(dataset),
        micro,
        world,
        rank,
        c["seed"],
        first=c["first_epoch_order"],
        later=c["later_epoch_order"],
        retain_tail=stage == "prompt",
    )
    epochs = cfg["stages"][stage + "_epochs"]
    micro_per_epoch = (
        math.ceil(len(dataset) / global_micro)
        if stage == "prompt"
        else len(dataset) // global_micro
    )
    if stage == "prompt" and accumulation != 1:
        raise ValueError(
            "prompt imitation uses a true partial tail; select micro_batch=effective_batch/world (accumulation=1)"
        )
    total_micro = epochs * micro_per_epoch
    if total_micro % accumulation:
        raise ValueError(
            "stage endpoint falls inside accumulation; choose matched batch geometry or an integer number of accumulation cycles"
        )
    total_updates = total_micro // accumulation
    contract = dict(
        stage=stage,
        config=digest(cfg),
        dataset=dataset.identity,
        features=provider.identity,
        micro_batch=micro,
        accumulation=accumulation,
        world_size=world,
        embedding=None if args.embedding is None else artifact_identity(args.embedding),
        init=None if not args.init else sha256(args.init),
        prompt_init=None if not args.prompt_init else sha256(args.prompt_init),
        prompt_init_selector=(
            (args.prompt_selector or cfg["stages"]["joint_prompt_selector"])
            if stage == "joint"
            else None
        ),
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(c["seed"] + rank)
    update = 0
    if args.resume:
        parent = load_tensor_file(args.resume)
        # Parent artifact paths need not be supplied again on an ordinary resume.
        contract["init"] = parent["contract"]["init"]
        contract["prompt_init"] = parent["contract"]["prompt_init"]
        contract["prompt_init_selector"] = parent["contract"]["prompt_init_selector"]
        update = restore_training(
            parent,
            contract=contract,
            models=models,
            emas=emas,
            optimizers=optimizers,
            cursor=cursor,
            balance=balance,
        )
    elif (out / "latest.pt").exists():
        raise FileExistsError(
            "output has a checkpoint; use --resume or a new output directory"
        )
    save = lambda name: save_training(
        out / name,
        cfg=cfg,
        contract=contract,
        models=models,
        emas=emas,
        optimizers=optimizers,
        cursor=cursor,
        update=update,
        balance=balance,
        extra=dict(
            total_updates=total_updates,
            stage_epochs=epochs,
            topology=dict(world=world, micro=micro, accumulation=accumulation),
        ),
    )
    if update == 0:
        save("initial.pt")
    requested = [False]

    def interrupt(signum, frame):
        requested[0] = True

    signal.signal(signal.SIGUSR1, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    forward = (
        torch.compile(elf)
        if elf is not None
        and c["compile"]
        and device.type == "cuda"
        and not args.no_compile
        else elf
    )
    if rank == 0:
        atomic_json(
            dict(
                contract=contract,
                planned_updates=total_updates,
                runtime=runtime_identity(),
                features=provider.identity,
                started=time.time(),
            ),
            out / "run.json",
        )
    limit = (
        min(total_updates, update + args.max_updates)
        if args.max_updates
        else total_updates
    )
    next_half = math.floor(update * accumulation / micro_per_epoch * 2) + 1
    while update < limit:
        step_start = time.monotonic()
        for oo in optimizers.values():
            for o in oo:
                o.zero_grad(set_to_none=True)
        values = []
        for _ in range(accumulation):
            indices, count = cursor.next()
            # Empty ranks at the prompt tail take a zero-weight real row so
            # collectives and parameter graphs still match the other ranks.
            local_count = len(indices)
            rows = [dataset[i] for i in indices] if indices else [dataset[0]]
            batch = collate(rows, cfg["data"]["max_tokens"], args.pad_id, device)
            target = provider(batch).detach().float()
            if stage == "prompt":
                with autocast(device, c["bf16"]):
                    prediction = encode_prompt(prompt, batch)
                loss = row_mse(prediction, target, batch["prompt"]) * (
                    world * local_count / count
                )
                (loss / accumulation).backward()
                metrics = dict(loss=float(loss.detach()), mse=float(loss.detach()))
            else:
                clean = target
                if prompt is not None:
                    with autocast(device, c["bf16"]):
                        condition = encode_prompt(prompt, batch)
                    clean = restore(target, condition, batch["prompt"])
                loss, metrics = mixed_loss(elf, clean, batch, cfg, forward=forward)
                (loss / accumulation).backward()
                if balance:
                    # Separate prompt-only pass, with unchanged training RNG streams.
                    with preserve_rng(), autocast(device, c["bf16"]):
                        mse = row_mse(
                            encode_prompt(prompt, batch), target, batch["prompt"]
                        )
                    balance.accumulate(mse, list(prompt.parameters()), accumulation)
                    metrics["prompt_mse"] = float(mse.detach())
            values.append(metrics)
        average_gradients(elf)
        balanced = {}
        if balance:
            balanced = {
                k: float(v)
                for k, v in balance.combine(
                    list(prompt.parameters()), accumulation
                ).items()
            }
        else:
            average_gradients(prompt)
        norms = {
            k: float(
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), c["clip_norm"], error_if_nonfinite=True
                )
            )
            for k, m in models.items()
        }
        for oo in optimizers.values():
            for o in oo:
                o.step()
        for k, m in models.items():
            emas[k].update(m)
        update += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        if rank == 0:
            metric = {
                k: sum(v.get(k, 0) for v in values) / len(values)
                for k in set().union(*(v.keys() for v in values))
            }
            append_json(
                dict(
                    update=update,
                    epoch=update * accumulation / micro_per_epoch,
                    seconds=time.monotonic() - step_start,
                    elapsed=time.monotonic() - start,
                    rank0_metrics=metric,
                    gradient_norms=norms,
                    **balanced,
                ),
                out / "metrics.jsonl",
            )
        flag = torch.tensor(int(requested[0]), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        half_due = update * accumulation >= next_half * micro_per_epoch / 2
        if half_due:
            save(f"half_epoch_{next_half:03d}.pt")
            next_half += 1
        if update % c["save_every"] == 0 or update == limit or flag.item():
            save("latest.pt")
        if flag.item():
            if rank == 0:
                atomic_json(
                    dict(status="interrupted", update=update, resume="latest.pt"),
                    out / "status.json",
                )
            return
    if update == total_updates:
        save("final.pt")
    if rank == 0:
        atomic_json(
            dict(
                status=(
                    "completed" if update == total_updates else "bounded_run_completed"
                ),
                updates=update,
                elapsed_seconds=time.monotonic() - start,
            ),
            out / "status.json",
        )
