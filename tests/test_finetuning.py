import copy
import json

import pytest
import torch
from safetensors.torch import load_file

from cedr.common import atomic_json, sha256
from cedr.finetuning import resolve, validate_data
from cedr.train import run
from cedr.data import Rows
from test_packages import package
from test_pipeline import fixture_files, args_for


def setup_package(tmp_path, cfg, rows, grouped=False):
    data, rep, embedding = fixture_files(tmp_path, cfg, rows)
    source = copy.deepcopy(cfg)
    source["model"].update(
        feature_group_time_conditioning=grouped,
        feature_group_time_names=["layer1", "layer2", "layer3"] if grouped else [],
    )
    root = package(tmp_path, source)
    payload = torch.load(rep, weights_only=True)
    payload.update(
        teacher={k: cfg["teacher"][k] for k in ("model_id", "revision")},
        compatible_models=["gsm8k-b-post-nft"],
    )
    torch.save(payload, rep)
    (root / "representation.pt").write_bytes(rep.read_bytes())
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(model_id="gsm8k-b-post-nft", representation="representation.pt")
    manifest["files"]["representation.pt"] = dict(bytes=rep.stat().st_size, sha256=sha256(rep))
    atomic_json(manifest, root / "manifest.json")
    cache = tmp_path / "cache/manifest.json"
    m = json.loads(cache.read_text())
    m["representation_sha256"] = sha256(rep)
    d = json.loads((data / "manifest.json").read_text())
    d.update(task=cfg["task"], teacher=cfg["teacher"], data=cfg["data"], inference=False)
    atomic_json(d, data / "manifest.json")
    m["dataset_sha256"] = sha256(data / "manifest.json")
    atomic_json(m, cache)
    args = args_for(tmp_path, data, embedding, "joint")
    args.embedding = None
    args.init_weights = str(root)
    return args, root


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("grouped", [False, True])
def test_weight_initialization_and_exact_resume(tmp_path, cfg, rows, grouped):
    args, root = setup_package(tmp_path, cfg, rows, grouped)
    cfg["training"]["prompt_mse_ratio"] = 2.0
    # Package architecture overrides recipe defaults, including time embedders.
    cfg["model"]["hidden_size"] = 64
    cfg["prompt"]["hidden_size"] = 64
    args.max_updates = 2
    run(args, cfg)
    initial = torch.load(tmp_path / "joint/initial.pt", weights_only=False)
    for component, filename in (("elf", "model.safetensors"), ("prompt", "prompt.safetensors")):
        source = load_file(root / filename)
        equal_tree(initial["models"][component], source)
        assert all(not o["state"] for o in initial["optimizers"][component])
        assert set(initial["emas"][component]) == {"0.99", "0.999", "0.9999"}
        for name in initial["emas"][component]["0.99"]:
            states = [initial["emas"][component][key][name] for key in ("0.99", "0.999", "0.9999")]
            assert len({v.data_ptr() for v in states}) == 3
            for value in states:
                equal_tree(value, source[name])
    assert initial["update"] == 0 and initial["balance"]["updates"] == 0
    assert initial["config"]["model"]["feature_group_time_conditioning"] == grouped
    assert initial["config"]["training"]["prompt_mse_ratio"] == 2.0
    args.init_weights = None
    args.resume = str(tmp_path / "joint/latest.pt")
    args.max_updates = None
    run(args, cfg)
    resumed = torch.load(tmp_path / "joint/final.pt", weights_only=False)
    args.resume = None
    args.init_weights = str(root)
    args.output = str(tmp_path / "uninterrupted")
    run(args, cfg)
    full = torch.load(tmp_path / "uninterrupted/final.pt", weights_only=False)
    for key in ("models", "emas", "optimizers", "scheduler", "cursor", "balance", "update"):
        equal_tree(resumed[key], full[key])
    metrics = [json.loads(line) for line in (tmp_path / "joint/metrics.jsonl").read_text().splitlines()]
    for component in ("elf", "prompt"):
        assert any(m["gradient_norms"][component] > 0 for m in metrics)
        assert all(torch.isfinite(torch.tensor(m["gradient_norms"][component])) for m in metrics)
    assert resumed["balance"]["updates"] == resumed["update"]


def test_initializer_rejects_conflicts_and_mismatches(tmp_path, cfg, rows):
    args, root = setup_package(tmp_path, cfg, rows)
    for key in ("init", "prompt_init", "resume", "elf_selector", "prompt_selector"):
        bad = copy.copy(args)
        setattr(bad, key, "wrong")
        with pytest.raises(ValueError, match="cannot be combined|already-selected"):
            resolve(bad, cfg)
    bad = copy.copy(args)
    bad.stage = "flow"
    with pytest.raises(ValueError, match="joint"):
        resolve(bad, cfg)
    bad = copy.copy(args)
    bad.representation = str(tmp_path / "wrong.pt")
    torch.save({}, bad.representation)
    with pytest.raises(ValueError, match="representation does not match"):
        resolve(bad, cfg)
    wrong_cfg = copy.deepcopy(cfg)
    wrong_cfg["teacher"]["revision"] = "wrong"
    with pytest.raises(ValueError, match="different teachers"):
        resolve(copy.copy(args), wrong_cfg)
    source, weights, parent, identity = resolve(copy.copy(args), cfg)
    data = Rows(args.data)
    data.manifest["teacher"] = dict(cfg["teacher"], revision="wrong")
    with pytest.raises(ValueError, match="tokenizer revision"):
        validate_data(data, source, root / "shared/tokenizer")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["representation"]
    atomic_json(manifest, manifest_path)
    with pytest.raises(ValueError, match="no representation"):
        resolve(copy.copy(args), cfg)


def test_representation_metadata_and_tokenizer_mismatch(tmp_path, cfg, rows, monkeypatch):
    import transformers
    from cedr.data import tokenizer_fingerprint

    args, root = setup_package(tmp_path, cfg, rows)
    class Tokenizer:
        special_tokens_map = {"eos_token": "end"}
        chat_template = "native"
        def get_vocab(self):
            return {"a": 1, "end": 2}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    data = Rows(args.data)
    data.manifest["tokenizer_sha256"] = tokenizer_fingerprint(Tokenizer())
    validate_data(data, cfg, root / "shared/tokenizer")
    data.manifest["tokenizer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="tokenizer differs"):
        validate_data(data, cfg, root / "shared/tokenizer")
    p = root / "representation.pt"
    payload = torch.load(p, weights_only=True)
    payload["compatible_models"] = ["wrong"]
    torch.save(payload, p)
    m = json.loads((root / "manifest.json").read_text())
    m["files"]["representation.pt"] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    atomic_json(m, root / "manifest.json")
    with pytest.raises(ValueError, match="metadata does not match"):
        resolve(copy.copy(args), cfg)
