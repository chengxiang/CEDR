import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from cedr.common import atomic_json, sha256
from cedr.packages import download
from test_packages import package


@pytest.mark.parametrize("with_representation", [False, True])
def test_selective_revision_pinned_download(tmp_path, cfg, monkeypatch, with_representation):
    import huggingface_hub

    source = package(tmp_path, cfg)
    model = "gsm8k-b-pre-nft"
    revision = "a" * 40
    manifest = json.loads((source / "manifest.json").read_text())
    if with_representation:
        representation = source / "representation.pt"
        representation.write_bytes(b"representation-download-fixture")
        manifest["representation"] = "representation.pt"
        manifest["files"]["representation.pt"] = dict(bytes=representation.stat().st_size, sha256=sha256(representation))
        atomic_json(manifest, source / "manifest.json")
    files = {
        **manifest["files"],
        "manifest.json": dict(
            bytes=(source / "manifest.json").stat().st_size,
            sha256=sha256(source / "manifest.json"),
        ),
    }
    index = dict(
        models={
            model: dict(
                files={
                    name: dict(
                        record,
                        path=(
                            name
                            if name.startswith("shared/")
                            else f"models/{model}/{name}"
                        ),
                    )
                    for name, record in files.items()
                }
            )
        }
    )
    atomic_json(index, tmp_path / "index.json")
    calls = []

    def get_file(repo, filename, **kwargs):
        assert repo == "xc91/CEDR"
        assert kwargs["revision"] == revision
        calls.append(filename)
        if filename == "index.json":
            return str(tmp_path / "index.json")
        return str(source / filename.removeprefix(f"models/{model}/"))

    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *a, **k: SimpleNamespace(sha=revision)
        ),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", get_file)
    result = download(model, tmp_path / "download")
    assert json.loads((result / "download.json").read_text())["revision"] == revision
    assert not any("post-nft" in path for path in calls)
    assert sha256(result / "model.safetensors") == sha256(source / "model.safetensors")
    if with_representation:
        assert sha256(result / "representation.pt") == sha256(source / "representation.pt")
    with pytest.raises(ValueError, match="unknown model"):
        download("other", tmp_path / "bad")
