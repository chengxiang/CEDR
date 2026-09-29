"""One feature contract for cached and frozen live-Qwen targets."""

from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .common import atomic_json, load_tensor_file, sha256, preserve_rng
from .data import collate


class Representation:
    def __init__(self, path, device="cpu"):
        self.identity = sha256(path)
        self.payload = load_tensor_file(path)
        if self.payload.get("format") != "cedr-representation-v1":
            raise ValueError("expected representation.pt from cedr projectors")
        self.layers = self.payload["layers"]
        self.dimensions = self.payload["dimensions"]
        self.device = torch.device(device)
        self.blocks = [
            (b["mean"].to(device).float(), b["encoder"].to(device).float())
            for b in self.payload["blocks"]
        ]
        for (_, e), d in zip(self.blocks, self.dimensions):
            if e.shape[1] != d:
                raise ValueError("representation shape mismatch")

    def project(self, hidden):
        with torch.autocast(self.device.type, enabled=False):
            return torch.cat(
                [
                    (hidden[l].float() - m) @ e
                    for l, (m, e) in zip(self.layers, self.blocks)
                ],
                -1,
            ).to(torch.bfloat16)


class LiveFeatures:
    def __init__(self, teacher, representation):
        self.teacher = teacher
        self.representation = representation
        self.identity = dict(
            backend="live",
            representation=representation.identity,
            teacher=teacher.identity,
        )

    @torch.no_grad()
    def __call__(self, batch):
        targets = []
        with preserve_rng():
            for lo in range(0, len(batch["rows"]), self.teacher.chunk_rows):
                hi = lo + self.teacher.chunk_rows
                sub = {
                    k: v[lo:hi] for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                hidden, _ = self.teacher.capture(sub)
                targets.append(self.representation.project(hidden))
        values = torch.cat(targets, 0)
        return values * batch["valid"].unsqueeze(-1)


class CachedFeatures:
    def __init__(self, path, dataset, representation=None):
        self.root = Path(path)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["dataset_sha256"] != dataset.identity:
            raise ValueError("cache/dataset identity mismatch")
        if representation and self.manifest["representation_sha256"] != sha256(
            representation
        ):
            raise ValueError("cache/representation mismatch")
        self.identity = dict(
            backend="cache",
            manifest=sha256(self.root / "manifest.json"),
            representation=self.manifest["representation_sha256"],
        )
        self.verified = set()

    def __call__(self, batch):
        out = torch.zeros(
            (*batch["ids"].shape, self.manifest["latent_dim"]),
            dtype=torch.bfloat16,
            device=batch["ids"].device,
        )
        shard_rows = self.manifest["shard_rows"]
        groups = {}
        for i, row in enumerate(batch["rows"]):
            groups.setdefault(row["physical_index"] // shard_rows, []).append((i, row))
        for shard, group in groups.items():
            record = self.manifest["shards"][shard]
            path = self.root / record["file"]
            if shard not in self.verified:
                if sha256(path) != record["sha256"]:
                    raise ValueError(f"corrupt feature shard: {path.name}")
                self.verified.add(shard)
            with safe_open(str(path), framework="pt", device="cpu") as f:
                offsets = f.get_tensor("offsets")
                features = f.get_slice("features")
                for i, row in group:
                    index = row["physical_index"] - record["first"]
                    start = int(offsets[index])
                    n = len(row["input_ids"])
                    if n > int(offsets[index + 1] - offsets[index]):
                        raise ValueError("feature/token length mismatch")
                    out[i, :n] = features[start : start + n].to(out.device)
        return out


def build_cache(
    dataset,
    provider,
    output,
    *,
    batch_rows=8,
    shard_rows=256,
    length=1024,
    pad_id=151643,
):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(
        version=1,
        dataset_sha256=dataset.identity,
        representation_sha256=provider.representation.identity,
        latent_dim=sum(provider.representation.dimensions),
        shard_rows=shard_rows,
        target_dtype="bfloat16",
    )
    if (root / "contract.json").exists():
        if json.loads((root / "contract.json").read_text()) != contract:
            raise ValueError("cache construction contract changed")
    else:
        atomic_json(contract, root / "contract.json")
    records = []
    for first in range(0, len(dataset), shard_rows):
        index = first // shard_rows
        path = root / f"features_{index:06d}.safetensors"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if not path.exists() or sha256(path) != record["sha256"]:
                raise ValueError("published shard changed")
            records.append(record)
            continue
        values = []
        offsets = [0]
        for lo in range(first, min(first + shard_rows, len(dataset)), batch_rows):
            rows = [
                dataset[i]
                for i in range(
                    lo, min(lo + batch_rows, first + shard_rows, len(dataset))
                )
            ]
            batch = collate(rows, length, pad_id, provider.teacher.device)
            targets = provider(batch)
            for row, value in zip(rows, targets):
                n = len(row["input_ids"])
                values.append(value[:n].cpu().contiguous())
                offsets.append(offsets[-1] + n)
        tmp = path.with_suffix(".tmp")
        save_file(
            dict(features=torch.cat(values), offsets=torch.tensor(offsets)), str(tmp)
        )
        tmp.replace(path)
        record = dict(
            file=path.name, sha256=sha256(path), first=first, rows=len(values)
        )
        atomic_json(record, receipt)
        records.append(record)
    result = dict(**contract, shards=records)
    atomic_json(result, root / "manifest.json")
    return result
