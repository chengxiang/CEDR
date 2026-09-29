import copy
import math
import pytest
import torch
from cedr.models import build_models, encode_prompt, promote_time_embedders, CEDR
from cedr.data import collate, Cursor
from cedr.objectives import (
    mixed_loss,
    draws,
    normalized_rewards,
    nft_numerator,
    nft_velocity,
)
from cedr.balance import PromptMSEBalance, row_mse
from cedr.state import EMA
from cedr.generation import grid, sample, decode
from cedr.representation import Covariance, make_projectors
from cedr.nn.decoder import chunked_factored_decoder_ce


def test_whitening_and_projector_gradient(cfg):
    torch.manual_seed(4)
    x = torch.randn(200, 16, dtype=torch.float64) @ torch.randn(
        16, 16, dtype=torch.float64
    )
    whole = Covariance(16)
    whole.update(x)
    split = Covariance(16)
    for part in x.split(37):
        split.update(part)
    torch.testing.assert_close(whole.mean, split.mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(whole.m2, split.m2, atol=1e-10, rtol=1e-12)
    stats = whole.export()
    p = make_projectors(
        dict(layers={i: stats for i in [1, 2, 3]}), [1, 2, 3], [4] * 3, "cpu"
    )["1"]
    e = p.effective_encoder().double()
    torch.testing.assert_close(
        e.T @ stats["covariance"] @ e,
        torch.eye(4, dtype=torch.float64),
        atol=2e-5,
        rtol=2e-5,
    )
    p(x.float()).square().mean().backward()
    assert p.raw_parameter.grad is not None and p.decoder.grad is not None


def test_chunked_ce_matches_full():
    torch.manual_seed(9)
    f = torch.randn(2, 7, 5, requires_grad=True)
    w = torch.randn(5, 13, requires_grad=True)
    b = torch.randn(13, requires_grad=True)
    ids = torch.randint(13, (2, 7))
    mask = torch.rand(2, 7) > 0.4
    actual, _, _ = chunked_factored_decoder_ce(f, ids, mask, w, b, chunk_tokens=3)
    expected = torch.nn.functional.cross_entropy(
        (f @ w + b)[mask], ids[mask], reduction="sum"
    )
    ga = torch.autograd.grad(actual, (f, w, b), retain_graph=True)
    ge = torch.autograd.grad(expected, (f, w, b))
    torch.testing.assert_close(actual, expected)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_prompt_only_frozen_and_independent_ema(cfg, rows):
    elf, prompt = build_models(cfg, "cpu", torch.randn(64, 16).bfloat16())
    batch = collate(rows[:2], 16, 0)
    a = encode_prompt(prompt, batch)
    batch["ids"][:, 3:] = 63
    b = encode_prompt(prompt, batch)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert "external_token_embedding" not in prompt.state_dict()
    a.sum().backward()
    assert prompt.external_token_embedding.grad is None
    ema = EMA(prompt, [0.99, 0.999, 0.9999])
    p = next(prompt.parameters())
    name = next(iter(dict(prompt.named_parameters())))
    assert (
        ema.values["0.99"][name].data_ptr()
        != ema.values["0.999"][name].data_ptr()
        != p.data_ptr()
    )
    initial = p.detach().clone()
    with torch.no_grad():
        p.add_(1)
    ema.update(prompt)
    torch.testing.assert_close(ema.values["0.99"][name], initial + 0.01)


def test_mixed_objective_and_routing(cfg, rows):
    torch.manual_seed(7)
    elf, prompt = build_models(cfg, "cpu", torch.randn(64, 16).bfloat16())
    batch = collate(rows[:2], 16, 0)
    target = torch.randn(2, 16, 12)
    condition = encode_prompt(prompt, batch)
    clean = torch.where(batch["prompt"][..., None], condition, target)
    random = draws(clean.shape, cfg, "cpu")
    random["decoder"] = torch.tensor([0.0, 1.0])
    random["use_sc"] = torch.ones(2)
    loss, m = mixed_loss(elf, clean, batch, cfg, random)
    loss.backward()
    assert m["decoder_rows"] == 1 and m["answer_tokens"] == 6 and torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in elf.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in prompt.parameters()
    )
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in elf.parameters())


def test_balance_reference_resume_and_zero():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    bal = PromptMSEBalance(target_ratio=2.0)
    ((p * torch.tensor([3.0, 4.0])).sum()).backward()
    mse = (p * torch.tensor([0.0, 2.0])).sum()
    bal.accumulate(mse, [p], 1)
    result = bal.combine([p], 1)
    torch.testing.assert_close(p.grad, torch.tensor([3.0, 14.0]))
    assert float(result["prompt_mse_achieved_ratio"]) == 2.0
    restored = PromptMSEBalance(target_ratio=2.0)
    restored.load_state_dict(bal.state_dict())
    assert restored.state_dict() == bal.state_dict()
    p.grad = torch.ones_like(p)
    restored.accumulate((p * 0).sum(), [p], 1)
    restored.combine([p], 1)
    assert torch.isfinite(p.grad).all()


def test_nft_rewards_and_normalization():
    correct = torch.tensor([[1, 1], [1, 0], [0, 0]])
    keep, rho = normalized_rewards(correct)
    assert keep.tolist() == [False, True, True]
    tiers = torch.tensor([[1.0, 0.75, 0.0], [1.0, 0.0, 0.0]])
    expected = (
        0.5
        + (
            (tiers - tiers.mean(1, keepdim=True))
            / (tiers.flatten().std(unbiased=False) + 1e-4)
        ).clamp(-5, 5)
        / 10
    )
    torch.testing.assert_close(rho, expected)
    cur = torch.randn(3, 4, 5, requires_grad=True)
    old = torch.randn_like(cur)
    ref = torch.randn_like(cur)
    data = torch.randn_like(cur)
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    r = torch.tensor([0.3, 0.9, 0.5])
    all_loss = nft_numerator(cur, old, ref, data, r, mask) / mask.sum()
    split = (
        sum(
            nft_numerator(
                cur[i : i + 1],
                old[i : i + 1],
                ref[i : i + 1],
                data[i : i + 1],
                r[i : i + 1],
                mask[i : i + 1],
            )
            for i in range(3)
        )
        / mask.sum()
    )
    torch.testing.assert_close(all_loss, split)


def test_clock_derivatives_and_terminal(cfg):
    class Oracle(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.ones_like(x[..., :12]), None

    cond = torch.zeros(2, 16, 12)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :3] = True
    g = torch.Generator().manual_seed(8)
    noise = torch.randn(cond.shape, generator=g)
    actual, _ = sample(
        Oracle(), cond, mask, cfg, generator=g, steps=4, initial_noise=noise
    )
    z = noise.clone()
    z[:, :3] = 0
    times = grid(4)
    for t, tn in zip(times[:-1], times[1:]):
        parts = []
        for v, p in zip(z.split(4, -1), [2.5, 2, 1.5]):
            parts.append(
                v + (tn - t) * p * t ** (p - 1) * (1 - v) / (1 - t**p).clamp_min(0.05)
            )
        z = torch.cat(parts, -1)
        z[:, :3] = 0
    torch.testing.assert_close(actual, z)


def test_scalar_equal_clocks_and_nft_promotion(cfg):
    torch.manual_seed(4)
    elf, _ = build_models(cfg, "cpu", stage="flow")
    x = torch.randn(2, 16, 24)
    t = torch.tensor([0.2, 0.4])
    g = torch.ones(2)
    a = elf(x, t, self_cond_cfg_scale=g)[0]
    b = elf(x, (t, t, t), self_cond_cfg_scale=g)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    options = {
        **cfg["model"],
        "feature_group_time_conditioning": True,
        "feature_group_time_names": ["layer1", "layer2", "layer3"],
    }
    nft = CEDR(**options)
    nft.load_state_dict(
        promote_time_embedders(elf.state_dict(), options["feature_group_time_names"])
    )
    torch.testing.assert_close(
        a, nft(x, (t, t, t), self_cond_cfg_scale=g)[0], atol=0, rtol=0
    )


def test_cursor_rank_coverage_tail_and_resume():
    a = Cursor(19, 2, 2, 0, 42)
    b = Cursor(19, 2, 2, 1, 42)
    seen = []
    for i in range(4):
        seen += a.next()[0] + b.next()[0]
    assert seen == list(range(16))
    assert a.state_dict() == b.state_dict()
    state = a.state_dict()
    expected = a.next()
    resume = Cursor(19, 2, 2, 0, 42)
    resume.load_state_dict(state)
    assert resume.next() == expected
    tail = [Cursor(7, 4, 2, r, 42, retain_tail=True) for r in range(2)]
    assert tail[0].next() == ([0, 1, 2, 3], 7)
    assert tail[1].next() == ([4, 5, 6], 7)
