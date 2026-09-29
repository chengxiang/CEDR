"""Small numerical, feature-extraction, training, and generation checks."""

from __future__ import annotations
import json
from pathlib import Path
import time
import torch
from .common import atomic_json, atomic_save, seed_all, sha256


def tiny_config(
    vocab=64,
    teacher_width=16,
    length=16,
    latent=12,
    layers=(1, 2, 3),
    dimensions=(4, 4, 4),
):
    return dict(
        task="gsm8k",
        teacher=dict(
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            chunk_rows=2,
        ),
        model=dict(
            text_encoder_dim=latent,
            max_length=length,
            hidden_size=64,
            depth=2,
            num_heads=4,
            bottleneck_dim=32,
            decoder_dim=32,
            num_time_tokens=4,
            num_self_cond_cfg_tokens=4,
            num_model_mode_tokens=4,
            vocab_size=vocab,
        ),
        prompt=dict(
            vocab_size=vocab,
            max_length=length,
            external_embedding_dim=teacher_width,
            bottleneck_dim=32,
            hidden_size=64,
            depth=2,
            num_heads=4,
            output_dim=latent,
        ),
        data=dict(max_tokens=length, max_prompt_tokens=length - 1),
        representation=dict(
            layers=list(layers),
            dimensions=list(dimensions),
            eigenvalue_floor=1e-5,
            projector_epochs=1,
            projector_lr=0.001,
            projector_ema=0.999,
        ),
        training=dict(
            seed=42,
            effective_batch=4,
            micro_batch=2,
            lr=0.002,
            clip_norm=1.0,
            ema_decays=[0.99, 0.999, 0.9999],
            bf16=True,
            compile=False,
            flow_mean=-1.5,
            flow_std=0.8,
            noise_scale=2.0,
            t_eps=0.05,
            decoder_prob=0.2,
            decoder_mean=0.8,
            decoder_std=0.8,
            decoder_noise_scale=1.0,
            self_cond_prob=0.5,
            sccfg_min=0.5,
            sccfg_max=5.0,
            ce_chunk_tokens=128,
            save_every=1,
            first_epoch_order="physical",
            later_epoch_order="global",
            prompt_mse_ratio=2.0,
            mse_balance_decay=0.99,
        ),
        stages=dict(
            flow_epochs=1,
            prompt_epochs=1,
            joint_epochs=1,
            joint_prompt_selector=".999",
            nft_updates=1,
        ),
        inference=dict(
            steps=4,
            clock_powers=[2.5, 2, 1.5],
            sccfg=2.0,
            cfg=1.0,
            grid_mean=-0.8,
            grid_std=0.8,
            elf_selector=".9999",
            prompt_selector=".9999",
        ),
        nft=dict(
            lr=0.0001,
            prompt_groups=2,
            generated_per_group=2,
            draws_per_endpoint=1,
            rollout_steps=2,
            rollout_sccfg=2.0,
            rollout_clock_powers=[1.0, 1.0, 1.0],
            beta=1.0,
            reference_weight=0.1,
            adv_clip_max=5.0,
            adv_epsilon=0.0001,
            old_transition_update=400,
            old_post_transition_decay=0.4,
            elf_init_selector=".9999",
            prompt_init_selector=".999",
            ema_decays=[0.9, 0.99, 0.999, 0.9999],
        ),
    )


def run(args):
    from .models import build_models, encode_prompt, make_optimizers
    from .data import Rows, collate, prepare
    from .features import Representation, LiveFeatures, CachedFeatures, build_cache
    from .teacher import QwenTeacher
    from .objectives import draws, mixed_loss, restore
    from .generation import sample, decode
    from .balance import PromptMSEBalance, row_mse
    from .state import EMA
    from .rewards import gsm_correct

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    seed_all(42)
    device = torch.device("cuda" if args.gpu else "cpu")
    checks = {}
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requires a GPU allocation")
        if not args.teacher or not args.representation:
            raise ValueError("GPU validation needs --teacher and --representation")
        from transformers import AutoTokenizer

        rep = Representation(args.representation, device)
        teacher = QwenTeacher(args.teacher, rep.layers, device, chunk_rows=2)
        tokenizer = AutoTokenizer.from_pretrained(args.teacher)
        cfg = tiny_config(
            vocab=teacher.embedding.shape[0],
            teacher_width=teacher.embedding.shape[1],
            length=1024,
            latent=sum(rep.dimensions),
            layers=rep.layers,
            dimensions=rep.dimensions,
        )
        panel = [
            dict(id=str(i), prompt=p, answer="One plus one is two. \\boxed{2}")
            for i, p in enumerate(
                [
                    "What is one plus one?",
                    "A child has one apple and receives another. How many apples?",
                    "Context: "
                    + "one apple and another apple. " * 100
                    + "\nWhat is one plus one?",
                    "Compute 1 + 1 and explain your reasoning.",
                ]
            )
        ]
        source = out / "panel.jsonl"
        source.write_text("".join(json.dumps(x) + "\n" for x in panel))
        prepare(source, out / "data", tokenizer, cfg)
        dataset = Rows(out / "data")
        rows = [dataset[i] for i in range(len(dataset))]
        batch = collate(rows, 1024, tokenizer.pad_token_id, device)
        assert max(r["prompt_length"] for r in rows) > 257
        live = LiveFeatures(teacher, rep)
        rng = torch.cuda.get_rng_state().clone()
        target = live(batch)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        checks["feature_rng_preserved"] = True
        build_cache(
            dataset,
            live,
            out / "cache",
            batch_rows=2,
            shard_rows=2,
            length=1024,
            pad_id=tokenizer.pad_token_id,
        )
        cached = CachedFeatures(out / "cache", dataset, args.representation)
        from_cache = cached(batch)
        torch.testing.assert_close(target, from_cache, atol=0, rtol=0)
        checks["cached_live_bitwise"] = True
        teacher.chunk_rows = 1
        different = live(batch)
        ratio = float(
            (different.float() - target.float()).square().mean().sqrt()
            / target.float().square().mean().sqrt().clamp_min(1e-12)
        )
        torch.testing.assert_close(
            different.float(), target.float(), atol=0.04, rtol=0.03
        )
        assert ratio < 0.01
        checks["chunk_change_relative_rms"] = ratio
        teacher.chunk_rows = 2
        sub = {
            k: v[:1] if isinstance(v, torch.Tensor) else v[:1] for k, v in batch.items()
        }
        hidden, _ = teacher.capture(sub)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = teacher.model.model(
                input_ids=sub["ids"],
                attention_mask=sub["valid"].long(),
                output_hidden_states=True,
                use_cache=False,
            )
        for layer in rep.layers:
            torch.testing.assert_close(
                hidden[layer], native.hidden_states[layer], atol=0, rtol=0
            )
        checks["native_prefix_parity"] = True
        changed = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sub.items()
        }
        changed["ids"][sub["answer"]] = tokenizer.pad_token_id
        second, _ = teacher.capture(changed)
        for layer in rep.layers:
            torch.testing.assert_close(
                hidden[layer][sub["prompt"]],
                second[layer][sub["prompt"]],
                atol=0,
                rtol=0,
            )
        checks["causal_prompt_independence"] = True
        replacement = hidden[rep.layers[0]].detach().float().requires_grad_()
        output = teacher.intervene(sub, rep.layers[0], replacement, sub["content"])
        output[sub["content"]].float().square().mean().backward()
        assert (
            replacement.grad is not None
            and torch.isfinite(replacement.grad).all()
            and replacement.grad.abs().sum() > 0
        )
        assert all(p.grad is None for p in teacher.model.parameters())
        checks["frozen_suffix_gradient"] = True
        embedding = teacher.embedding
        checks["teacher_snapshot"] = str(args.teacher)
        checks["representation_sha256"] = sha256(args.representation)
    else:
        cfg = tiny_config()
        cfg["training"]["bf16"] = False
        rows = [
            dict(
                id=str(i), input_ids=[1, 2, 3, 4, 5, 6], prompt_length=3, content_end=5
            )
            for i in range(4)
        ]
        batch = collate(rows, 16, 0)
        target = torch.randn(4, 16, 12).bfloat16() * batch["valid"][..., None]
        from_cache = target.clone()
        embedding = torch.randn(64, 16).bfloat16()
    elf, prompt = build_models(cfg, device, embedding)
    opt = make_optimizers(elf, prompt, 0.002)
    emas = {
        "elf": EMA(elf, [0.99, 0.999, 0.9999]),
        "prompt": EMA(prompt, [0.99, 0.999, 0.9999]),
    }
    forward = torch.compile(elf) if args.gpu else elf
    random = draws(target.shape, cfg, device)
    random["decoder"] = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    # Compare feature and loss parity under an identical realization of every draw.
    la, _ = mixed_loss(elf, target.float(), batch, cfg, random, forward=forward)
    lb, _ = mixed_loss(elf, from_cache.float(), batch, cfg, random, forward=forward)
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    checks["cached_live_loss_bitwise"] = True
    del la, lb
    bal = PromptMSEBalance(target_ratio=2.0)
    losses = []
    for step in range(2):
        for oo in opt.values():
            for o in oo:
                o.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.gpu):
            condition = encode_prompt(prompt, batch)
        clean = restore(target.float(), condition, batch["prompt"])
        loss, metrics = mixed_loss(elf, clean, batch, cfg, random, forward=forward)
        loss.backward()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.gpu):
            aux = row_mse(encode_prompt(prompt, batch), target.float(), batch["prompt"])
        bal.accumulate(aux, list(prompt.parameters()), 1)
        balance = bal.combine(list(prompt.parameters()), 1)
        for m in (elf, prompt):
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        for oo in opt.values():
            for o in oo:
                o.step()
        emas["elf"].update(elf)
        emas["prompt"].update(prompt)
        losses.append(
            {
                **metrics,
                "prompt_mse": float(aux.detach()),
                "lambda": float(balance["prompt_mse_lambda"]),
            }
        )
    assert prompt.external_token_embedding.grad is None
    checks["two_balanced_joint_updates"] = losses
    checks["frozen_embedding"] = True
    gen = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        condition = encode_prompt(prompt, batch, True)
    z, counts = sample(elf, condition, batch["prompt"], cfg, generator=gen, steps=4)
    ids = decode(elf, z, 2.0)
    assert ids.shape == batch["ids"].shape and torch.isfinite(z).all()
    checks["generation"] = dict(**counts, decoder_calls=1, shape=list(ids.shape))
    assert gsm_correct("The answer is 2.", "#### 2") and not gsm_correct(
        "The answer is 3.", "#### 2"
    )
    checks["canonical_scoring_controls"] = True
    atomic_save(
        dict(
            models={"elf": elf.state_dict(), "prompt": prompt.state_dict()},
            emas={k: e.state_dict() for k, e in emas.items()},
            balance=bal.state_dict(),
        ),
        out / "validation_checkpoint.pt",
    )
    if args.gpu:
        torch.cuda.synchronize()
    atomic_json(
        dict(
            status="passed",
            seconds=time.monotonic() - start,
            device=str(device),
            checks=checks,
        ),
        out / "validation.json",
    )
    print(
        json.dumps(
            dict(
                status="passed",
                seconds=time.monotonic() - start,
                report=str(out / "validation.json"),
            ),
            indent=2,
        )
    )
