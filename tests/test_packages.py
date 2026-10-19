import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from cedr.common import atomic_json, sha256
from cedr.packages import read_package, safe_path, load_package
from cedr.models import build_models
from cedr.cli import parser
from test_pipeline import fixture_files


def package(tmp_path, cfg):
    root = tmp_path / "model"
    (root / "shared/tokenizer").mkdir(parents=True)
    embedding = torch.randn(64, 16).bfloat16()
    elf, prompt = build_models(cfg, "cpu", embedding)
    for name, state in [
        ("model.safetensors", elf.state_dict()),
        ("prompt.safetensors", prompt.state_dict()),
        ("shared/embedding.safetensors", {"embedding": embedding}),
    ]:
        save_file(state, str(root / name))
    cfg["inference"]["steps"] = 3
    cfg["inference"]["max_response_tokens"] = 4
    atomic_json(cfg, root / "config.json")
    files = {
        name: dict(bytes=(root / name).stat().st_size, sha256=sha256(root / name))
        for name in (
            "model.safetensors",
            "prompt.safetensors",
            "shared/embedding.safetensors",
            "config.json",
        )
    }
    manifest = dict(
        format="cedr-inference-v2",
        config="config.json",
        files=files,
        weights=dict(elf="model.safetensors", prompt="prompt.safetensors"),
        embedding="shared/embedding.safetensors",
        tokenizer="shared/tokenizer",
        selection=dict(elf="0.9999", prompt="0.999"),
    )
    atomic_json(manifest, root / "manifest.json")
    return root


@pytest.mark.parametrize("format_version", [1, 2])
def test_package_integrity_and_paths(tmp_path, cfg, format_version):
    root = package(tmp_path, cfg)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["format"] = f"cedr-inference-v{format_version}"
    atomic_json(manifest, root / "manifest.json")
    ck, manifest = load_package(root)
    assert ck["selection"] == dict(elf="0.9999", prompt="0.999")
    assert len(ck["models"]["elf"]) > 0
    for name in ("../escape", "/absolute", "x/../../y"):
        with pytest.raises(ValueError):
            safe_path(root, name)
    with (root / "prompt.safetensors").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="verification"):
        read_package(root)


@pytest.mark.parametrize("sampler", ["ode", "edm", "dmd2"])
def test_package_generation_exact_resume(tmp_path, cfg, rows, monkeypatch, sampler):
    cfg["inference"]["sampler"] = sampler
    from cedr import generation
    import transformers

    class Tokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, name):
            return 63

        def decode(self, tokens, **kwargs):
            return ",".join(map(str, tokens))

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    monkeypatch.setattr(
        generation, "distributed_setup", lambda: (0, 1, torch.device("cpu"))
    )
    data, _, _ = fixture_files(tmp_path, cfg, rows[:5])
    root = package(tmp_path, cfg)

    def args(out):
        return parser().parse_args(
            [
                "generate",
                "--checkpoint",
                str(root),
                "--data",
                str(data),
                "--output",
                str(out),
                "--batch-size",
                "2",
            ]
        )

    generation.run(args(tmp_path / "full"), cfg)
    original = generation.atomic_save

    def interrupt(value, path):
        original(value, path)
        raise InterruptedError("simulated interruption after durable batch")

    monkeypatch.setattr(generation, "atomic_save", interrupt)
    with pytest.raises(InterruptedError):
        generation.run(args(tmp_path / "resumed"), cfg)
    monkeypatch.setattr(generation, "atomic_save", original)
    generation.run(args(tmp_path / "resumed"), cfg)
    assert (tmp_path / "full/predictions.jsonl").read_bytes() == (
        tmp_path / "resumed/predictions.jsonl"
    ).read_bytes()
    assert (
        json.loads((tmp_path / "resumed/generation.json").read_text())["coverage"] == 5
    )
    assert all(
        len(json.loads(line)["token_ids"]) <= 4
        for line in (tmp_path / "resumed/predictions.jsonl").read_text().splitlines()
    )
    changed = args(tmp_path / "resumed")
    changed.seed = 123
    with pytest.raises(ValueError, match="settings changed"):
        generation.run(changed, cfg)
    changed = args(tmp_path / "bad-selector")
    changed.prompt_selector = ".99"
    with pytest.raises(ValueError, match="contains only"):
        generation.run(changed, cfg)
