"""Gold-anchored NFT with task rewards and synchronous training clocks."""

from __future__ import annotations
import copy
from pathlib import Path
import time
import signal
import numpy as np
import torch
import torch.distributed as dist
from .common import (
    runtime_identity,
    artifact_identity,
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
)
from .data import Rows, Cursor, collate
from .models import build_models, encode_prompt, make_optimizers, promote_time_embedders
from .teacher import load_embedding
from .state import EMA, save_training, restore_training, selected_state
from .objectives import normalized_rewards, nft_numerator, nft_velocity, velocity, draws
from .generation import sample, decode
from .rewards import reward

DECODER_ONLY = {
    "mode_tokens",
    "proj_kernel",
    "proj_bias",
    "unembed_kernel",
    "unembed_bias",
}


@torch.no_grad()
def update_old(old, current, decay):
    for a, b in zip(old.parameters(), current.parameters(), strict=True):
        a.mul_(decay).add_(b, alpha=1 - decay)


def run(args, cfg):
    from transformers import AutoTokenizer
    from .train import feature_provider

    cfg = copy.deepcopy(cfg)
    cfg["model"]["feature_group_time_conditioning"] = True
    cfg["model"]["feature_group_time_names"] = [
        f"layer{layer}" for layer in cfg["representation"]["layers"]
    ]
    rank, world, device = distributed_setup()
    c = cfg["training"]
    n = cfg["nft"]
    start = time.monotonic()
    if not args.init and not args.resume:
        raise ValueError("NFT requires --init supervised checkpoint")
    if cfg["task"] == "oci" and not args.reward_validation:
        raise ValueError(
            "OCI requires --reward-validation receipt from reference validation"
        )
    data = Rows(args.data)
    if cfg["task"] == "oci":
        import json

        receipt = json.loads(Path(args.reward_validation).read_text())
        if receipt["eligible_sha256"] != data.manifest["input_sha256"]:
            raise ValueError("OCI data differs from reference-validated population")
    provider = feature_provider(args, cfg, data, device)
    with preserve_rng():
        seed_all(c["seed"])
        elf, prompt = build_models(cfg, device, load_embedding(args.embedding))
    if not args.resume:
        source = load_tensor_file(args.init)
        elf.load_state_dict(
            promote_time_embedders(
                selected_state(
                    source, "elf", args.elf_selector or n["elf_init_selector"]
                ),
                cfg["model"]["feature_group_time_names"],
            ),
            strict=True,
        )
        prompt.load_state_dict(
            selected_state(
                source, "prompt", args.prompt_selector or n["prompt_init_selector"]
            ),
            strict=True,
        )
    for name, p in elf.named_parameters():
        if name in DECODER_ONLY:
            p.requires_grad_(False)
    old_elf = copy.deepcopy(elf).eval().requires_grad_(False)
    old_prompt = copy.deepcopy(prompt).eval().requires_grad_(False)
    ref_elf = copy.deepcopy(elf).eval().requires_grad_(False)
    ref_prompt = copy.deepcopy(prompt).eval().requires_grad_(False)
    models = dict(
        elf=elf,
        prompt=prompt,
        old_elf=old_elf,
        old_prompt=old_prompt,
        ref_elf=ref_elf,
        ref_prompt=ref_prompt,
    )
    optimizers = make_optimizers(elf, prompt, n["lr"])
    emas = {k: EMA(m, n["ema_decays"]) for k, m in dict(elf=elf, prompt=prompt).items()}
    groups = n["prompt_groups"]
    if groups % world:
        raise ValueError("prompt_groups must be divisible by the torchrun world size")
    unique = np.load(Path(args.data) / "prompts.npy", mmap_mode="r")
    cursor = Cursor(
        len(unique),
        groups // world,
        world,
        rank,
        c["seed"],
        first="global",
        later="global",
    )
    contract = dict(
        stage="nft",
        config=digest(cfg),
        dataset=data.identity,
        features=provider.identity,
        embedding=artifact_identity(args.embedding),
        world_size=world,
        train_micro=args.micro_batch or c["micro_batch"],
        init=sha256(args.init) if args.init else None,
        initialization_selectors=dict(
            elf=args.elf_selector or n["elf_init_selector"],
            prompt=args.prompt_selector or n["prompt_init_selector"],
        ),
        reward_validation=(
            sha256(args.reward_validation) if args.reward_validation else None
        ),
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(c["seed"] + rank)
    update = 0
    rounds = 0
    gen = torch.Generator(device=device).manual_seed(c["seed"] + rank)
    if args.resume:
        saved = load_tensor_file(args.resume)
        contract["init"] = saved["contract"]["init"]
        contract["initialization_selectors"] = saved["contract"][
            "initialization_selectors"
        ]
        update = restore_training(
            saved,
            contract=contract,
            models=models,
            emas=emas,
            optimizers=optimizers,
            cursor=cursor,
        )
        gen.set_state(saved["extra"]["rollout_rng"][rank].cpu())
        rounds = saved["extra"]["rounds"]
    elif (out / "latest.pt").exists():
        raise FileExistsError("existing NFT checkpoint; pass --resume")

    def save(name):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, gen.get_state())
        else:
            states[0] = gen.get_state()
        save_training(
            out / name,
            cfg=cfg,
            contract=contract,
            models=models,
            emas=emas,
            optimizers=optimizers,
            cursor=cursor,
            update=update,
            extra=dict(
                rounds=rounds,
                rollout_rng=states,
                initialization_selectors=contract["initialization_selectors"],
            ),
        )

    if update == 0:
        save("initial.pt")
    current_forward = (
        torch.compile(elf)
        if c["compile"] and device.type == "cuda" and not args.no_compile
        else elf
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or cfg["teacher"]["model_id"],
        revision=cfg["teacher"]["revision"],
    )
    terminal = tokenizer.convert_tokens_to_ids("<|im_end|>")
    K = n["generated_per_group"]
    width = cfg["model"]["text_encoder_dim"]
    length = cfg["data"]["max_tokens"]
    total = cfg["stages"]["nft_updates"]
    limit = min(total, update + args.max_updates) if args.max_updates else total
    requested = [False]

    def interrupt(signum, frame):
        requested[0] = True

    signal.signal(signal.SIGUSR1, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if rank == 0:
        atomic_json(
            dict(
                contract=contract,
                total_updates=total,
                runtime=runtime_identity(),
                auxiliary_prompt_mse=False,
            ),
            out / "run.json",
        )
    while update < limit:
        step_start = time.monotonic()
        indices, _ = cursor.next()
        rows = [data[int(unique[i])] for i in indices]
        rounds += 1
        gold_batch = collate(rows, length, tokenizer.pad_token_id, device)
        gold = provider(gold_batch).detach().float()
        endpoints = []
        endpoint_masks = []
        endpoint_rows = []
        local_correct = []
        with torch.no_grad():
            for i, row in enumerate(rows):
                repeated = [row] * K
                batch = collate(repeated, length, tokenizer.pad_token_id, device)
                condition = encode_prompt(old_prompt, batch, True)
                z, _ = sample(
                    old_elf,
                    condition,
                    batch["prompt"],
                    cfg,
                    generator=gen,
                    steps=n["rollout_steps"],
                    powers=n["rollout_clock_powers"],
                    sccfg=n["rollout_sccfg"],
                )
                tokens = decode(old_elf, z, n["rollout_sccfg"])
                correct = []
                masks = []
                for ids in tokens.cpu().tolist():
                    answer = ids[row["prompt_length"] :]
                    end = (
                        answer.index(terminal) + 1
                        if terminal in answer
                        else len(answer)
                    )
                    text = tokenizer.decode(answer[:end], skip_special_tokens=True)
                    correct.append(reward(cfg["task"], text, row))
                    mask = torch.zeros(length, dtype=torch.bool, device=device)
                    mask[row["prompt_length"] : row["prompt_length"] + end] = True
                    masks.append(mask)
                local_correct.append(correct)
                endpoints.append(torch.cat([gold[i : i + 1], z], 0))
                endpoint_masks.append(
                    torch.cat([gold_batch["answer"][i : i + 1], torch.stack(masks)], 0)
                )
                endpoint_rows.extend([row] * (K + 1))
        correct = torch.tensor(local_correct, dtype=torch.bool, device=device)
        if world > 1:
            gathered = [torch.empty_like(correct) for _ in range(world)]
            dist.all_gather(gathered, correct)
            all_correct = torch.cat(gathered)
        else:
            all_correct = correct
        keep, coefficients = normalized_rewards(
            all_correct, n["adv_epsilon"], n["adv_clip_max"]
        )
        # Scatter normalized coefficients back without including discarded groups in std.
        full_rho = torch.zeros((groups, K + 1), device=device)
        full_rho[keep] = coefficients
        local_keep = keep[rank * len(rows) : (rank + 1) * len(rows)]
        local_rho = full_rho[rank * len(rows) : (rank + 1) * len(rows)]
        if not bool(keep.any()):
            if rank == 0:
                append_json(
                    dict(round=rounds, update=update, discard_fraction=1.0),
                    out / "metrics.jsonl",
                )
            flag = torch.tensor(int(requested[0]), device=device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                save("latest.pt")
                return
            if rounds > 1000 + update * 100:
                raise RuntimeError(
                    "too many fully discarded rounds; inspect the training population"
                )
            continue
        x = torch.cat(endpoints, 0)
        masks = torch.cat(endpoint_masks, 0)
        rho = local_rho.flatten()
        active = local_keep.repeat_interleave(K + 1)
        endpoint_ids = (
            active.nonzero().flatten().repeat_interleave(n["draws_per_endpoint"])
        )
        denom = masks[active].sum() * n["draws_per_endpoint"]
        denom = denom.double()
        if world > 1:
            dist.all_reduce(denom)
        if denom <= 0:
            raise ValueError("retained NFT endpoints have zero scored tokens")
        for oo in optimizers.values():
            for o in oo:
                o.zero_grad(set_to_none=True)
        loss_value = 0.0
        micro = args.micro_batch or c["micro_batch"]
        # Every rank may retain a different number of records. No DDP collectives
        # occur inside this loop; gradients are summed once after all local work.
        for lo in range(0, len(endpoint_ids), micro):
            chosen = endpoint_ids[lo : lo + micro]
            rr = [endpoint_rows[int(i)] for i in chosen.cpu()]
            batch = collate(rr, length, tokenizer.pad_token_id, device)
            target = x[chosen]
            mask = masks[chosen]
            random = draws(target.shape, cfg, device)
            t = random["t"]
            z = t[:, None, None] * target + (1 - t[:, None, None]) * random["noise"]
            with torch.no_grad():
                old_condition = encode_prompt(old_prompt, batch, True)
                ref_condition = encode_prompt(ref_prompt, batch, True)
                old_v = nft_velocity(
                    old_elf,
                    z,
                    old_condition,
                    batch["prompt"],
                    t,
                    random["use_sc"],
                    random["guidance"],
                    bf16=c["bf16"],
                )
                ref_v = nft_velocity(
                    ref_elf,
                    z,
                    ref_condition,
                    batch["prompt"],
                    t,
                    random["use_sc"],
                    random["guidance"],
                    bf16=c["bf16"],
                )
            with autocast(device, c["bf16"]):
                condition = encode_prompt(prompt, batch)
            current = nft_velocity(
                current_forward,
                z,
                condition,
                batch["prompt"],
                t,
                random["use_sc"],
                random["guidance"],
                bf16=c["bf16"],
            )
            data_v = velocity(target, z, t, c["t_eps"])
            numerator = nft_numerator(
                current,
                old_v,
                ref_v,
                data_v,
                rho[chosen],
                mask,
                beta=n["beta"],
                clip=n["adv_clip_max"],
                reference_weight=n["reference_weight"],
            )
            # average_gradients divides by world, cancel it to obtain an UPDATE-
            # wide token normalization even when retained counts differ by rank.
            loss = numerator * (world / denom.float())
            loss.backward()
            loss_value += float((numerator / denom).detach())
        average_gradients(elf)
        average_gradients(prompt)
        norms = {
            k: float(
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), c["clip_norm"], error_if_nonfinite=True
                )
            )
            for k, m in dict(elf=elf, prompt=prompt).items()
        }
        for oo in optimizers.values():
            for o in oo:
                o.step()
        for k, m in dict(elf=elf, prompt=prompt).items():
            emas[k].update(m)
        update += 1
        decay = (
            min(0.001 * update, 0.5)
            if update < n["old_transition_update"]
            else n["old_post_transition_decay"]
        )
        update_old(old_elf, elf, decay)
        update_old(old_prompt, prompt, decay)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if rank == 0:
            append_json(
                dict(
                    update=update,
                    round=rounds,
                    seconds=time.monotonic() - step_start,
                    elapsed=time.monotonic() - start,
                    rank0_loss_contribution=loss_value,
                    global_answer_tokens=float(denom),
                    discard_fraction=float((~keep).float().mean()),
                    generated_correct=float(all_correct.float().mean()),
                    old_policy_decay=decay,
                    gradient_norms=norms,
                ),
                out / "metrics.jsonl",
            )
        flag = torch.tensor(int(requested[0]), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if update % c["save_every"] == 0 or update == limit or flag.item():
            save("latest.pt")
        if update % 100 == 0:
            save(f"update_{update:04d}.pt")
        if flag.item():
            return
    if update == total:
        save("final.pt")
    if rank == 0:
        atomic_json(
            dict(
                status="completed" if update == total else "bounded_run_completed",
                updates=update,
                rounds=rounds,
            ),
            out / "status.json",
        )
