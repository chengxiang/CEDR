"""Mixed flow/decoder objective and NFT numerical kernels."""

from __future__ import annotations
import math
import torch
from .nn.decoder import chunked_factored_decoder_ce
from .common import autocast


def restore(x, prompt, mask):
    return torch.where(mask.unsqueeze(-1), prompt, x)


def velocity(pred, z, t, eps=0.05):
    return (pred.float() - z.float()) / (1 - t.float()).clamp_min(eps)[:, None, None]


def draws(shape, cfg, device):
    b, s, d = shape
    c = cfg["training"]
    return dict(
        t=torch.sigmoid(c["flow_mean"] + c["flow_std"] * torch.randn(b, device=device)),
        noise=torch.randn(shape, device=device) * c["noise_scale"],
        decoder=(torch.rand(b, device=device) < c["decoder_prob"]).float(),
        decoder_lambda=torch.sigmoid(
            c["decoder_mean"] + c["decoder_std"] * torch.randn((b, s, 1), device=device)
        ),
        decoder_noise=torch.randn(shape, device=device) * c["decoder_noise_scale"],
        use_sc=(torch.rand(b, device=device) < c["self_cond_prob"]).float(),
        guidance=(1 + c["sccfg_min"])
        * torch.exp(
            torch.rand(b, device=device)
            * math.log((1 + c["sccfg_max"]) / (1 + c["sccfg_min"]))
        )
        - 1,
    )


def mixed_loss(model, clean, batch, cfg, randomness=None, forward=None):
    c = cfg["training"]
    r = draws(clean.shape, cfg, clean.device) if randomness is None else randomness
    forward = model if forward is None else forward
    t = r["t"]
    p = batch["prompt"]
    answer = batch["answer"]
    eps = c["t_eps"]
    z = restore(
        t[:, None, None] * clean + (1 - t[:, None, None]) * r["noise"], clean, p
    )
    # The detached auxiliary passes cannot update either model or prompt encoder.
    with torch.no_grad(), autocast(clean.device, c["bf16"]):
        zero = restore(torch.zeros_like(z), clean, p)
        pred0 = forward(
            torch.cat([z, zero], -1),
            t,
            deterministic=True,
            self_cond_cfg_scale=r["guidance"],
        )[0]
        pred1 = forward(
            torch.cat([z, restore(pred0, clean, p)], -1),
            t,
            deterministic=True,
            self_cond_cfg_scale=r["guidance"],
        )[0]
        target = velocity(clean, z, t, eps) + (r["use_sc"] * (1 - 1 / r["guidance"]))[
            :, None, None
        ] * (velocity(pred1, z, t, eps) - velocity(pred0, z, t, eps))
    decoder_z = restore(
        r["decoder_lambda"] * clean + (1 - r["decoder_lambda"]) * r["decoder_noise"],
        clean,
        p,
    )
    active = r["decoder"][:, None, None]
    mixed_z = active * decoder_z + (1 - active) * z
    self_condition = restore(r["use_sc"][:, None, None] * pred0.detach(), clean, p) * (
        1 - active
    )
    mixed_t = r["decoder"] + (1 - r["decoder"]) * t
    with autocast(clean.device, c["bf16"]):
        pred, features = forward(
            torch.cat([mixed_z, self_condition], -1),
            mixed_t,
            deterministic=False,
            self_cond_cfg_scale=r["guidance"],
            decoder_step_active=r["decoder"],
            decoder_return_features=True,
        )
    flow = (velocity(pred, z, t, eps) - target.detach()).square().mean(-1)
    flow_sum = (flow * answer * (1 - r["decoder"][:, None])).sum()
    ce_sum, _, _ = chunked_factored_decoder_ce(
        features,
        batch["ids"],
        answer & r["decoder"][:, None].bool(),
        model.unembed_kernel,
        model.unembed_bias,
        chunk_tokens=c["ce_chunk_tokens"],
    )
    count = answer.sum()
    if count == 0:
        raise ValueError("flow/CE batch has no answer targets")
    loss = (flow_sum + ce_sum) / count
    return loss, dict(
        loss=float(loss.detach()),
        flow=float((flow_sum / count).detach()),
        ce=float((ce_sum / count).detach()),
        answer_tokens=int(count),
        decoder_rows=int(r["decoder"].sum()),
    )


def normalized_rewards(correct, epsilon=1e-4, clip=5.0):
    if (
        correct.ndim != 2
        or correct.shape[1] < 1
        or not bool(((correct == 0) | (correct == 1)).all())
    ):
        raise ValueError("correctness must be binary [groups,K]")
    keep = ~correct.bool().all(1)
    tiers = torch.cat(
        [torch.ones_like(correct[:, :1], dtype=torch.float32), correct.float() * 0.75],
        1,
    )[keep]
    if tiers.numel() == 0:
        return keep, tiers
    centered = tiers - tiers.mean(1, keepdim=True)
    advantage = (centered / (tiers.flatten().std(unbiased=False) + epsilon)).clamp(
        -clip, clip
    )
    return keep, (0.5 + advantage / (2 * clip)).clamp(0, 1)


def nft_numerator(
    current,
    old,
    reference,
    data,
    rho,
    mask,
    *,
    beta=1.0,
    clip=5.0,
    reference_weight=0.1,
):
    """Gold-anchored NFT loss numerator with reference-policy regularization."""
    residual = current.float() - data.float()
    old_residual = old.detach().float() - data.float()
    pos = (1 - beta) * old_residual + beta * residual
    neg = (1 + beta) * old_residual - beta * residual
    policy = (clip / beta) * (
        rho[:, None] * pos.square().mean(-1)
        + (1 - rho[:, None]) * neg.square().mean(-1)
    )
    ref = (current.float() - reference.detach().float()).square().mean(-1)
    return ((policy + reference_weight * ref) * mask).sum()


def nft_velocity(
    model, z, clean_prompt, prompt_mask, t, use_sc, guidance, *, bf16=True, eps=0.05
):
    z = restore(z, clean_prompt, prompt_mask)
    with torch.no_grad(), autocast(z.device, bf16):
        zero = restore(torch.zeros_like(z), clean_prompt, prompt_mask)
        p0 = model(
            torch.cat([z, zero], -1),
            t,
            deterministic=True,
            self_cond_cfg_scale=guidance,
        )[0]
    sc = restore(p0.detach() * use_sc[:, None, None], clean_prompt, prompt_mask)
    with autocast(z.device, bf16):
        pred = model(
            torch.cat([z, sc], -1), t, deterministic=True, self_cond_cfg_scale=guidance
        )[0]
    v0 = velocity(p0, z, t, eps)
    vm = velocity(pred, z, t, eps)
    correction = (use_sc * (1 - 1 / guidance))[:, None, None] * (
        vm.detach() - v0.detach()
    )
    return vm - correction
