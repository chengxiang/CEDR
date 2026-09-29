import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from cedr.common import atomic_json, atomic_save, sha256, rng_state
from cedr.data import Rows, collate, encode_row
from cedr.features import LiveFeatures, CachedFeatures, Representation, build_cache
from cedr.train import run


class ToyTeacher:
    device = torch.device("cpu")
    layers = (1, 2, 3)
    chunk_rows = 2
    identity = {"fixture": "causal"}

    def capture(self, batch, final=False):
        h = batch["ids"].float().cumsum(1)[:, :, None].expand(-1, -1, 16) / 100
        return {i: h + i for i in self.layers}, h


def fixture_files(tmp_path, cfg, rows):
    data = tmp_path / "data"
    data.mkdir()
    offsets = []
    with (data / "rows.jsonl").open("wb") as f:
        for row in rows:
            offsets.append(f.tell())
            f.write((json.dumps(row) + "\n").encode())
    np.save(data / "offsets.npy", offsets)
    np.save(data / "prompts.npy", np.arange(len(rows)))
    atomic_json(
        dict(
            files={
                p: sha256(data / p)
                for p in ["rows.jsonl", "offsets.npy", "prompts.npy"]
            }
        ),
        data / "manifest.json",
    )
    rep = tmp_path / "rep.pt"
    atomic_save(
        dict(
            format="cedr-representation-v1",
            layers=[1, 2, 3],
            dimensions=[4] * 3,
            blocks=[
                dict(mean=torch.zeros(16), encoder=torch.eye(16)[:, :4])
                for _ in range(3)
            ],
        ),
        rep,
    )
    provider = LiveFeatures(ToyTeacher(), Representation(rep))
    dataset = Rows(data)
    build_cache(
        dataset,
        provider,
        tmp_path / "cache",
        batch_rows=2,
        shard_rows=3,
        length=16,
        pad_id=0,
    )
    embedding = tmp_path / "embedding.safetensors"
    save_file({"embedding": torch.randn(64, 16).bfloat16()}, str(embedding))
    return data, rep, embedding


def test_cache_live_mask_rng_and_loss(tmp_path, cfg, rows):
    data, rep, embedding = fixture_files(tmp_path, cfg, rows)
    dataset = Rows(data)
    batch = collate([dataset[i] for i in [4, 1, 7]], 16, 0)
    teacher = ToyTeacher()
    live = LiveFeatures(teacher, Representation(rep))
    cached = CachedFeatures(tmp_path / "cache", dataset, rep)
    before = torch.get_rng_state().clone()
    a = live(batch)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(a, cached(batch), rtol=0, atol=0)
    teacher.chunk_rows = 1
    torch.testing.assert_close(a, live(batch), rtol=0, atol=0)
    from cedr.models import build_models
    from cedr.objectives import draws, mixed_loss

    elf, _ = build_models(cfg, "cpu", stage="flow")
    r = draws(a.shape, cfg, "cpu")
    la, _ = mixed_loss(elf, a.float(), batch, cfg, r)
    lb, _ = mixed_loss(elf, cached(batch).float(), batch, cfg, r)
    torch.testing.assert_close(la, lb, rtol=0, atol=0)


def args_for(tmp_path, data, embedding, stage):
    return SimpleNamespace(
        stage=stage,
        data=str(data),
        features="cache",
        cache=str(tmp_path / "cache"),
        representation=None,
        teacher=None,
        qwen_chunk=None,
        embedding=str(embedding),
        init=None,
        prompt_init=None,
        resume=None,
        elf_selector=None,
        prompt_selector=None,
        micro_batch=None,
        max_updates=None,
        no_compile=True,
        pad_id=0,
        output=str(tmp_path / stage),
        tokenizer=None,
        reward_validation=None,
    )


def test_training_resume_prompt_joint_and_export(tmp_path, cfg, rows):
    data, rep, embedding = fixture_files(tmp_path, cfg, rows)
    a = args_for(tmp_path, data, embedding, "flow")
    a.max_updates = 2
    run(a, cfg)
    a.resume = str(tmp_path / "flow/latest.pt")
    a.max_updates = None
    run(a, cfg)
    actual = torch.load(tmp_path / "flow/final.pt", weights_only=False)
    a.resume = None
    a.output = str(tmp_path / "uninterrupted")
    run(a, cfg)
    expected = torch.load(tmp_path / "uninterrupted/final.pt", weights_only=False)
    for key in expected["models"]["elf"]:
        torch.testing.assert_close(
            actual["models"]["elf"][key], expected["models"]["elf"][key], atol=0, rtol=0
        )
    p = args_for(tmp_path, data, embedding, "prompt")
    run(p, cfg)
    j = args_for(tmp_path, data, embedding, "joint")
    j.init = str(tmp_path / "flow/final.pt")
    j.prompt_init = str(tmp_path / "prompt/final.pt")
    j.max_updates = 1
    cfg["training"]["prompt_mse_ratio"] = 2.0
    run(j, cfg)
    initial = torch.load(tmp_path / "joint/initial.pt", weights_only=False)
    latest = torch.load(tmp_path / "joint/latest.pt", weights_only=False)
    for key in initial["models"]["elf"]:
        torch.testing.assert_close(
            initial["models"]["elf"][key], actual["models"]["elf"][key], rtol=0, atol=0
        )
    assert latest["balance"]["updates"] == 1
    assert not initial["optimizers"]["prompt"][0]["state"]
    from cedr.state import export_checkpoint, selected_state

    export_checkpoint(
        tmp_path / "joint/latest.pt", tmp_path / "weights.pt", ".9999", ".999"
    )
    result = torch.load(tmp_path / "weights.pt", weights_only=False)
    assert result["format"] == "cedr-export-v1" and "optimizers" not in result
    assert result["stage"] == "joint"
    assert result["selection"] == dict(elf=".9999", prompt=".999")
    for component, selector in [("elf", ".9999"), ("prompt", ".999")]:
        selected = selected_state(latest, component, selector)
        assert result["models"][component].keys() == selected.keys()
        for name, value in selected.items():
            torch.testing.assert_close(
                result["models"][component][name], value, rtol=0, atol=0
            )
    export_checkpoint(tmp_path / "joint/latest.pt", tmp_path / "all-weights.pt")
    all_states = torch.load(tmp_path / "all-weights.pt", weights_only=False)
    for component in latest["models"]:
        for selector in ["raw", ".99", ".999", ".9999"]:
            selected = selected_state(all_states, component, selector)
            for name, value in selected_state(latest, component, selector).items():
                torch.testing.assert_close(selected[name], value, rtol=0, atol=0)


def test_tiny_qwen_causal_and_suffix_gradient(cfg, rows):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from cedr.teacher import QwenTeacher

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    t = QwenTeacher.__new__(QwenTeacher)
    t.device = torch.device("cpu")
    t.layers = (1, 2, 3)
    t.chunk_rows = 2
    t.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    batch = collate(rows[:2], 16, 0)
    a, last = t.capture(batch, final=True)
    t.chunk_rows = 1
    chunked, chunked_last = t.capture(batch, final=True)
    for layer in a:
        torch.testing.assert_close(a[layer], chunked[layer], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(last, chunked_last, atol=1e-5, rtol=1e-4)
    batch2 = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    batch2["ids"][:, 3:6] = 10
    b, _ = t.capture(batch2)
    for l in a:
        torch.testing.assert_close(a[l][:, :3], b[l][:, :3], atol=0, rtol=0)
    replacement = a[2].detach().float().requires_grad_()
    changed = t.intervene(batch, 2, replacement, batch["content"])
    changed.float().square().sum().backward()
    assert replacement.grad[:, 3:5].abs().sum() > 0
    assert replacement.grad[:, :3].abs().sum() == 0 and all(
        p.grad is None for p in t.model.parameters()
    )


def test_scoring_refuses_changed_predictions(tmp_path, cfg, rows):
    from cedr.scoring import score

    data, _, _ = fixture_files(tmp_path, cfg, rows)
    predictions = tmp_path / "predictions.jsonl"
    values = [dict(id=r["id"], prediction="\\boxed{2}", seed=42) for r in rows]
    predictions.write_text("".join(json.dumps(v) + "\n" for v in values))
    args = SimpleNamespace(
        data=data,
        predictions=predictions,
        output=tmp_path / "scoring",
        benchmark=None,
        evalplus_cache=None,
        workers=1,
    )
    score(args, cfg)
    report = json.loads((args.output / "scoring.json").read_text())
    assert report["correct"] == len(rows)
    values[0]["prediction"] = "\\boxed{3}"
    predictions.write_text("".join(json.dumps(v) + "\n" for v in values))
    with pytest.raises(ValueError, match="scoring inputs changed"):
        score(args, cfg)


def test_nft_end_to_end_and_resume(tmp_path, cfg, rows, monkeypatch):
    data, rep, embedding = fixture_files(tmp_path, cfg, rows)
    from cedr.models import build_models
    from cedr.state import EMA

    elf, prompt = build_models(cfg, "cpu", torch.randn(64, 16).bfloat16())
    atomic_save(
        dict(
            config=cfg,
            models={"elf": elf.state_dict(), "prompt": prompt.state_dict()},
            emas={
                "elf": EMA(elf, [0.99, 0.999, 0.9999]).state_dict(),
                "prompt": EMA(prompt, [0.99, 0.999, 0.9999]).state_dict(),
            },
        ),
        tmp_path / "source.pt",
    )

    class Tokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, x):
            return 6

        def decode(self, *args, **kwargs):
            return "The answer is 3."

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    a = args_for(tmp_path, data, embedding, "nft")
    a.init = str(tmp_path / "source.pt")
    a.micro_batch = 2
    cfg["stages"]["nft_updates"] = 2
    a.max_updates = 1
    run(a, cfg)
    initial = torch.load(tmp_path / "nft/initial.pt", weights_only=False)
    first = torch.load(tmp_path / "nft/latest.pt", weights_only=False)
    assert (
        first["update"] == 1
        and first["extra"]["rounds"] == 1
        and first["balance"] is None
    )
    for key in (
        "proj_kernel",
        "proj_bias",
        "unembed_kernel",
        "unembed_bias",
        "mode_tokens",
    ):
        torch.testing.assert_close(
            first["models"]["elf"][key], initial["models"]["elf"][key], rtol=0, atol=0
        )
    a.resume = str(tmp_path / "nft/latest.pt")
    a.max_updates = None
    a.init = None
    run(a, cfg)
    resumed = torch.load(tmp_path / "nft/final.pt", weights_only=False)
    a.resume = None
    a.init = str(tmp_path / "source.pt")
    a.output = str(tmp_path / "nft_control")
    run(a, cfg)
    control = torch.load(tmp_path / "nft_control/final.pt", weights_only=False)
    for component in control["models"]:
        for key in control["models"][component]:
            torch.testing.assert_close(
                resumed["models"][component][key],
                control["models"][component][key],
                rtol=0,
                atol=0,
            )


def test_projector_live_prepared_and_resume(tmp_path, cfg, rows):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from cedr.teacher import QwenTeacher
    from cedr.representation import train_projectors, prepare_activations

    data, rep, embedding = fixture_files(tmp_path, cfg, rows)
    t = QwenTeacher.__new__(QwenTeacher)
    t.device = torch.device("cpu")
    t.layers = (1, 2, 3)
    t.chunk_rows = 2
    t.identity = {"fixture": "tiny-qwen"}
    t.model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    stats = dict(
        mean=torch.zeros(32, dtype=torch.float64),
        eigenvalues=torch.ones(32, dtype=torch.float64),
        eigenvectors=torch.eye(32, dtype=torch.float64),
        hidden_whitener=torch.eye(32, dtype=torch.float64),
        hidden_sqrt=torch.eye(32, dtype=torch.float64),
        eigenvalue_floor=1e-5,
    )
    cov = tmp_path / "cov.pt"
    atomic_save(dict(layers={i: stats for i in (1, 2, 3)}), cov)
    ds = Rows(data)
    cfg["representation"]["projector_epochs"] = 1
    train_projectors(
        ds, t, cov, tmp_path / "live-projectors", cfg, batch_rows=2, max_updates=1
    )
    prepare_activations(
        ds, t, tmp_path / "activations", batch_rows=2, length=16, pad_id=0
    )
    train_projectors(
        ds,
        t,
        cov,
        tmp_path / "prepared-projectors",
        cfg,
        batch_rows=2,
        activation_cache=tmp_path / "activations",
        max_updates=1,
    )
    a = torch.load(tmp_path / "live-projectors/latest.pt", weights_only=False)
    b = torch.load(tmp_path / "prepared-projectors/latest.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    train_projectors(
        ds,
        t,
        cov,
        tmp_path / "live-projectors",
        cfg,
        batch_rows=2,
        resume=tmp_path / "live-projectors/latest.pt",
    )
    result = torch.load(
        tmp_path / "live-projectors/representation.pt", weights_only=False
    )
    assert result["source"]["updates"] == 4


def test_stratified_panel_weights(tmp_path):
    from cedr.selection import select

    rows = [
        dict(
            id=str(i),
            prompt="p",
            answer="a",
            metadata={"subject": "a" if i < 8 else "b"},
        )
        for i in range(10)
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    select(source, tmp_path / "panel.jsonl", 4, ["subject"], 42)
    values = [
        json.loads(s) for s in (tmp_path / "panel.jsonl").read_text().splitlines()
    ]
    assert [v["metadata"]["subject"] for v in values].count("a") == 2
    assert sum(v["weight"] for v in values) == 10
    select(source, tmp_path / "panel2.jsonl", 4, ["subject"], 42)
    assert (tmp_path / "panel.jsonl").read_bytes() == (
        tmp_path / "panel2.jsonl"
    ).read_bytes()
