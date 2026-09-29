"""Streaming covariance and independent teacher-soft-CE projector learning."""

from __future__ import annotations
from pathlib import Path
import json
import torch
from .common import (
    atomic_save,
    atomic_json,
    append_json,
    sha256,
    load_tensor_file,
    seed_all,
    autocast,
)
from .data import collate
from .nn.projector import DiffQRReconstructionBottleneck
from .optim import muon_with_aux_adam


class Covariance:
    """Weighted, mergeable FP64 centered moments (avoids subtractive cancellation)."""

    def __init__(self, width, device="cpu"):
        self.weight = 0.0
        self.mean = torch.zeros(width, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(width, width, dtype=torch.float64, device=device)

    def update(self, x, weight=1.0):
        x = x.double()
        if x.numel() == 0:
            return
        w = float(weight) * len(x)
        if w <= 0:
            raise ValueError("covariance weights must be positive")
        mean = x.mean(0)
        centered = x - mean
        delta = mean - self.mean
        total = self.weight + w
        self.m2.add_(centered.T @ centered, alpha=float(weight)).add_(
            torch.outer(delta, delta), alpha=self.weight * w / total
        )
        self.mean.add_(delta, alpha=w / total)
        self.weight = total

    def export(self, floor=1e-5):
        if self.weight <= 0:
            raise ValueError("empty covariance")
        covariance = self.m2 / self.weight
        eig, u = torch.linalg.eigh(covariance)
        eig = eig.flip(0)
        u = u.flip(1)
        # Keep eigensolver signs/order in the saved artifact. Every consumer reuses them.
        regular = eig.clamp_min(floor)
        return dict(
            mean=self.mean.cpu(),
            covariance=covariance.cpu(),
            eigenvalues=eig.cpu(),
            eigenvectors=u.cpu(),
            hidden_whitener=((u * regular.rsqrt()) @ u.T).cpu(),
            hidden_sqrt=((u * regular.sqrt()) @ u.T).cpu(),
            weight=self.weight,
            eigenvalue_floor=floor,
        )


def estimate(
    dataset, teacher, output, *, batch_rows=4, length=1024, pad_id=151643, floor=1e-5
):
    stats = {
        l: Covariance(teacher.embedding.shape[1], teacher.device)
        for l in teacher.layers
    }
    for lo in range(0, len(dataset), batch_rows):
        rows = [dataset[i] for i in range(lo, min(lo + batch_rows, len(dataset)))]
        batch = collate(rows, length, pad_id, teacher.device)
        activations, _ = teacher.capture(batch)
        for i, row in enumerate(rows):
            for l in teacher.layers:
                stats[l].update(activations[l][i, batch["content"][i]], row["weight"])
    result = dict(
        format="cedr-covariance-v1",
        dataset_sha256=dataset.identity,
        teacher=teacher.identity,
        layers={l: s.export(floor) for l, s in stats.items()},
    )
    atomic_save(result, output)
    return result


def make_projectors(stats, layers, dimensions, device):
    branches = torch.nn.ModuleDict()
    for layer, width in zip(layers, dimensions):
        s = stats["layers"][layer]
        e = s["eigenvalues"].clamp_min(s["eigenvalue_floor"])
        pca = s["eigenvectors"][:, :width] * e[:width].rsqrt()
        branches[str(layer)] = DiffQRReconstructionBottleneck(
            layer=layer,
            mean=s["mean"],
            hidden_whitener=s["hidden_whitener"],
            hidden_sqrt=s["hidden_sqrt"],
            pca_effective_encoder=pca,
        ).to(device)
    return branches


def export_projectors(branches, output, *, source, dimensions):
    layers = [int(k) for k in branches]
    payload = dict(
        format="cedr-representation-v1",
        layers=layers,
        dimensions=dimensions,
        source=source,
        blocks=[
            dict(
                mean=branches[str(l)].mean.cpu(),
                encoder=branches[str(l)].effective_encoder().detach().cpu(),
                decoder=branches[str(l)].decoder.detach().cpu(),
            )
            for l in layers
        ],
    )
    atomic_save(payload, output)
    return payload


def prepare_activations(
    dataset, teacher, output, *, batch_rows=4, length=1024, pad_id=151643
):
    """Optional row-aligned activations; no flattened-token teacher alignment ambiguity."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    contract = dict(
        dataset_sha256=dataset.identity, teacher=teacher.identity, batch_rows=batch_rows
    )
    if (out / "contract.json").exists() and json.loads(
        (out / "contract.json").read_text()
    ) != contract:
        raise ValueError("activation contract changed")
    atomic_json(contract, out / "contract.json")
    for lo in range(0, len(dataset), batch_rows):
        path = out / f"{lo:09d}.pt"
        if not path.exists():
            rows = [dataset[i] for i in range(lo, min(lo + batch_rows, len(dataset)))]
            batch = collate(rows, length, pad_id, teacher.device)
            hs, last = teacher.capture(batch, final=True)
            atomic_save(
                dict(
                    ids=[r["id"] for r in rows],
                    layers={k: v.cpu() for k, v in hs.items()},
                    final=last.cpu(),
                ),
                path,
            )
        records.append(dict(first=lo, file=path.name, sha256=sha256(path)))
    atomic_json(dict(**contract, records=records), out / "manifest.json")


def train_projectors(
    dataset,
    teacher,
    covariance,
    output,
    cfg,
    *,
    batch_rows=4,
    activation_cache=None,
    resume=None,
    max_updates=None,
):
    rc = cfg["representation"]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    stats = load_tensor_file(covariance)
    branches = make_projectors(stats, rc["layers"], rc["dimensions"], teacher.device)
    optimizers = {
        k: muon_with_aux_adam(b, rc["projector_lr"]) for k, b in branches.items()
    }
    from .state import EMA

    ema = EMA(branches, [0.99, 0.999, 0.9999])
    epoch = 0
    cursor = 0
    updates = 0
    seed_all(cfg["training"]["seed"])
    contract = dict(
        dataset=dataset.identity,
        covariance=sha256(covariance),
        recipe=rc,
        batch_rows=batch_rows,
    )
    if activation_cache:
        ac = Path(activation_cache)
        manifest = json.loads((ac / "manifest.json").read_text())
        if (
            manifest["dataset_sha256"] != dataset.identity
            or manifest["batch_rows"] != batch_rows
        ):
            raise ValueError("activation cache alignment mismatch")
    if resume:
        ck = load_tensor_file(resume)
        if ck["contract"] != contract:
            raise ValueError("projector continuation contract changed")
        branches.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        for k, o in optimizers.items():
            o.load_state_dict(ck["optimizers"][k])
        epoch, cursor, updates = ck["epoch"], ck["cursor"], ck["updates"]
    while epoch < rc["projector_epochs"]:
        for lo in range(cursor, len(dataset), batch_rows):
            rows = [dataset[i] for i in range(lo, min(lo + batch_rows, len(dataset)))]
            batch = collate(
                rows,
                cfg["data"]["max_tokens"],
                cfg["data"].get("pad_token_id", 151643),
                teacher.device,
            )
            if activation_cache:
                rec = manifest["records"][lo // batch_rows]
                p = ac / rec["file"]
                if sha256(p) != rec["sha256"]:
                    raise ValueError("activation cache changed")
                cached = load_tensor_file(p)
                if cached["ids"] != [r["id"] for r in rows]:
                    raise ValueError("activation row identities differ")
                hs = {k: v.to(teacher.device) for k, v in cached["layers"].items()}
                last = cached["final"].to(teacher.device)
            else:
                hs, last = teacher.capture(batch, final=True)
            # Hidden j predicts token j+1. Exclude first assistant target and terminal target.
            selection = batch["content"] & torch.roll(batch["content"], -1, 1)
            selection[:, -1] = False
            indices = selection.flatten().nonzero().flatten()
            if indices.numel() == 0:
                raise ValueError("projector batch has no adjacent assistant tokens")
            weights = (
                torch.tensor([r["weight"] for r in rows], device=teacher.device)[
                    :, None
                ]
                .expand_as(selection)
                .flatten()[indices]
            )
            denominator = weights.sum()
            metrics = {}
            # Independent branches: never feed one reconstruction into the next branch.
            for key, branch in branches.items():
                optimizers[key].zero_grad(set_to_none=True)
                with torch.autocast(teacher.device.type, enabled=False):
                    replacement = branch(hs[int(key)].float())
                modified = teacher.intervene(
                    batch, int(key), replacement, batch["content"]
                )
                loss = torch.zeros((), device=teacher.device)
                clean_flat = last.flatten(0, 1)
                changed_flat = modified.flatten(0, 1)
                for begin in range(0, len(indices), 256):
                    ii = indices[begin : begin + 256]
                    with torch.no_grad():
                        prob = teacher.logits(clean_flat[ii]).softmax(-1)
                    ce = -(prob * teacher.logits(changed_flat[ii]).log_softmax(-1)).sum(
                        -1
                    )
                    loss = (
                        loss + (ce * weights[begin : begin + 256]).sum() / denominator
                    )
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    branch.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizers[key].step()
                metrics[key] = dict(soft_ce=float(loss.detach()), grad_norm=float(norm))
            ema.update(branches)
            updates += 1
            cursor = lo + len(rows)
            append_json(
                dict(epoch=epoch, rows=cursor, updates=updates, layers=metrics),
                out / "metrics.jsonl",
            )
            if (
                updates % 100 == 0
                or cursor == len(dataset)
                or (max_updates and updates >= max_updates)
            ):
                atomic_save(
                    dict(
                        contract=contract,
                        model=branches.state_dict(),
                        ema=ema.state_dict(),
                        optimizers={k: o.state_dict() for k, o in optimizers.items()},
                        epoch=epoch,
                        cursor=cursor,
                        updates=updates,
                    ),
                    out / "latest.pt",
                )
            if max_updates and updates >= max_updates:
                break
        if max_updates and updates >= max_updates:
            break
        epoch += 1
        cursor = 0
    with ema.selected(branches, str(rc["projector_ema"])):
        export_projectors(
            branches,
            out / "representation.pt",
            source=dict(
                contract=contract,
                updates=updates,
                epochs_complete=epoch,
                ema=rc["projector_ema"],
            ),
            dimensions=rc["dimensions"],
        )
