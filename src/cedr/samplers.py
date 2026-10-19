"""Clean-prediction re-noising and EDM-style quantile-index churn."""

import math
from statistics import NormalDist
import torch
from .common import autocast
from .objectives import restore


def noise_seed(seed, first_row):
    return (int(seed) + 1_000_003 + int(first_row) * 1_000_003) % (2**63 - 1)


def quantile(probability, mean, std):
    if not 0 <= probability <= 1 or not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("expected a probability and finite logit-normal parameters")
    if probability in (0, 1):
        return float(probability)
    x = mean + std * NormalDist().inv_cdf(probability)
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def edm_plan(times, powers, c, sigma, mean, std):
    """Contract the quantile index, rather than multiplying the stored time."""
    if not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError("edm_churn must be finite and in [0,1]")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("noise scale must be finite and positive")
    steps = len(times) - 1
    if steps < 1 or times[0] != 0 or times[-1] != 1:
        raise ValueError("expected a complete zero-to-one grid")
    if any(abs(t - quantile(i / steps, mean, std)) > 2e-6 for i, t in enumerate(times)):
        raise ValueError("EDM-style churn requires the logit-normal quantile grid")
    plans = []
    for i, (t, tn) in enumerate(zip(times[:-1], times[1:])):
        if not 0 <= t < tn <= 1:
            raise ValueError("grid must be strictly increasing")
        interior = i > 0 and tn < 1
        back = quantile(c * i / steps, mean, std) if interior and c != 1 else t
        local = [t ** p for p in powers]
        tb = [back ** p for p in powers]
        dest = [tn ** p for p in powers]
        alpha = [b / a if interior else 1.0 for a, b in zip(local, tb)]
        variance = [sigma ** 2 * (1 - a) * (1 + a - 2 * b) for a, b in zip(alpha, tb)]
        if not 0 <= back <= t or min(variance) < -1e-12:
            raise ValueError("invalid backward transition")
        plans.append(dict(
            evaluation_tau=back,
            alpha=alpha,
            variance=[max(v, 0.0) for v in variance],
            weights=[(d - b) / (1 - b) for d, b in zip(dest, tb)],
            noise_applied=interior and c < 1,
            terminal=tn == 1,
        ))
    return plans


def guided_clean(model, z, sc, local, condition, mask, scale, cfg_scale):
    pred = model(torch.cat([z, sc], -1), local, deterministic=True,
                 self_cond_cfg_scale=scale)[0]
    calls = 1
    if cfg_scale != 1:
        zeros = torch.zeros_like(condition)
        uncond = model(torch.cat([restore(z, zeros, mask), restore(sc, zeros, mask)], -1),
                       local, deterministic=True, self_cond_cfg_scale=scale)[0]
        pred = uncond + cfg_scale * (pred - uncond)
        calls += 1
    return restore(pred, condition, mask), calls


@torch.no_grad()
def sample_stochastic(model, condition, mask, cfg, *, generator, sampler,
                      steps=None, powers=None, sccfg=None, cfg_scale=1.0,
                      bf16=False, initial_noise=None, renoise_eta=None,
                      edm_churn=None, batch_offset=0):
    from .generation import grid

    if sampler not in ("dmd2", "edm"):
        raise ValueError("sampler must be ode, edm, or dmd2")
    infer = cfg["inference"]
    steps = infer["steps"] if steps is None else steps
    powers = infer["clock_powers"] if powers is None else powers
    dims = cfg["representation"]["dimensions"]
    if len(powers) != len(dims) or any(not math.isfinite(p) or p < 1 for p in powers):
        raise ValueError("one finite clock exponent >=1 is required per feature group")
    if sum(dims) != condition.shape[-1]:
        raise ValueError("clock/latent dimensions differ")
    eta = infer.get("renoise_eta", 1.0) if renoise_eta is None else renoise_eta
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("renoise_eta must be finite and nonnegative")
    c = infer.get("edm_churn", 0.7) if edm_churn is None else edm_churn
    sigma = cfg["training"]["noise_scale"]
    z = (torch.randn(condition.shape, device=generator.device, generator=generator,
                     dtype=torch.float32).to(condition.device) * sigma
         if initial_noise is None else initial_noise.clone()).float()
    z = restore(z, condition, mask)
    sc = restore(torch.zeros_like(z), condition, mask)
    times = grid(steps, infer["grid_mean"], infer["grid_std"], z.device)
    gammas = torch.tensor(powers, device=z.device, dtype=torch.float32).unbind()
    plans = edm_plan(times.cpu().tolist(), [float(g) for g in gammas], c, sigma,
                     infer["grid_mean"], infer["grid_std"]) if sampler == "edm" else None
    fresh_seed = noise_seed(generator.initial_seed(), batch_offset)
    fresh = torch.Generator(device=z.device).manual_seed(fresh_seed)
    scale = torch.full((len(z),), float(infer["sccfg"] if sccfg is None else sccfg), device=z.device)
    calls = draws = applied = 0
    with autocast(z.device, bf16):
        for i, (current, following) in enumerate(zip(times[:-1], times[1:])):
            t, tn = current.item(), following.item()
            plan = plans[i] if plans is not None else None
            noise = None
            if plan is not None and not plan["terminal"]:
                # Consume the initial draw even though the initial transition has no churn.
                noise = torch.randn(z.shape, generator=fresh, device=z.device, dtype=z.dtype)
                draws += 1
                if plan["noise_applied"]:
                    z = torch.cat([a * part + math.sqrt(v) * eps for part, eps, a, v in
                                   zip(z.split(dims, -1), noise.split(dims, -1), plan["alpha"], plan["variance"])], -1)
                    z = restore(z, condition, mask)
                    applied += 1
            tau = torch.full((len(z),), plan["evaluation_tau"] if plan else t,
                             device=z.device, dtype=torch.float32)
            local = tuple(tau.pow(g) for g in gammas)
            clean, n = guided_clean(model, z, sc, local, condition, mask, scale, cfg_scale)
            calls += n
            if tn == 1:
                z = restore(clean.to(z.dtype), condition, mask)
            elif sampler == "dmd2":
                noise = torch.randn(z.shape, generator=fresh, device=z.device, dtype=z.dtype) * (float(eta) * float(sigma))
                draws += 1
                applied += int(eta > 0)
                next_tau = torch.full_like(tau, tn)
                weights = [next_tau.pow(g).reshape(-1, 1, 1).to(z.dtype) for g in gammas]
                z = torch.cat([w * p.to(z.dtype) + (1 - w) * e for w, p, e in
                               zip(weights, clean.split(dims, -1), noise.split(dims, -1))], -1)
                z = restore(z, condition, mask)
            else:
                z = torch.cat([part + w * (pred.float() - part) for part, pred, w in
                               zip(z.split(dims, -1), clean.split(dims, -1), plan["weights"])], -1)
                z = restore(z, condition, mask)
            sc = clean
    if not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite stochastic terminal state")
    return z, dict(denoiser_calls=calls, steps=steps, sampler=sampler,
                   terminal_latent="clean_prediction", noise_seed=fresh_seed,
                   noise_generator=str(z.device), noise_draws=draws,
                   applied_noise_intervals=applied,
                   **({"renoise_eta": float(eta)} if sampler == "dmd2" else {"edm_churn": float(c)}))
