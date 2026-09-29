"""Independent EMAs and atomic, optimizer-boundary training checkpoints."""

from __future__ import annotations
from contextlib import contextmanager
import torch
import torch.distributed as dist
from .common import (
    atomic_save,
    rng_state,
    set_rng,
    runtime_identity,
    load_tensor_file,
    sha256,
)


def decay_key(value):
    return format(float(value), ".12g")


class EMA:
    def __init__(self, module, decays):
        self.values = {
            decay_key(d): {k: p.detach().clone() for k, p in module.named_parameters()}
            for d in decays
        }

    @torch.no_grad()
    def update(self, module):
        params = dict(module.named_parameters())
        for key, state in self.values.items():
            for name, value in state.items():
                value.mul_(float(key)).add_(params[name].detach(), alpha=1 - float(key))

    def state_dict(self):
        return self.values

    def load_state_dict(self, states):
        if set(states) != set(self.values):
            raise ValueError("EMA decays changed")
        for key, state in states.items():
            if set(state) != set(self.values[key]):
                raise ValueError("EMA parameter coverage changed")
            for name, value in state.items():
                self.values[key][name].copy_(value)

    @contextmanager
    def selected(self, module, selector):
        if selector == "raw":
            yield
            return
        key = decay_key(selector)
        raw = {k: p.detach().clone() for k, p in module.named_parameters()}
        with torch.no_grad():
            for k, p in module.named_parameters():
                p.copy_(self.values[key][k])
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in module.named_parameters():
                    p.copy_(raw[k])


def selected_state(bundle, component, selector):
    if selector == "raw":
        return bundle["models"][component]
    key = decay_key(selector)
    if key not in bundle["emas"][component]:
        raise ValueError(f"{component} EMA {key} is unavailable")
    return {**bundle["models"][component], **bundle["emas"][component][key]}


def save_training(
    path,
    *,
    cfg,
    contract,
    models,
    emas,
    optimizers,
    cursor,
    update,
    balance=None,
    extra=None,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local = rng_state()
    ranks = [None] * world
    if world > 1:
        dist.all_gather_object(ranks, local)
    else:
        ranks[0] = local
    if rank == 0:
        atomic_save(
            dict(
                format="cedr-training-v1",
                runtime=runtime_identity(),
                config=cfg,
                contract=contract,
                world_size=world,
                models={k: m.state_dict() for k, m in models.items() if m is not None},
                emas={k: e.state_dict() for k, e in emas.items()},
                optimizers={
                    k: [o.state_dict() for o in oo] for k, oo in optimizers.items()
                },
                scheduler=dict(kind="constant", update=update),
                cursor=cursor.state_dict(),
                update=update,
                rank_rng=ranks,
                balance=None if balance is None else balance.state_dict(),
                extra=extra or {},
            ),
            path,
        )
    if world > 1:
        dist.barrier()


def restore_training(
    bundle, *, contract, models, emas, optimizers, cursor, balance=None
):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if (
        bundle["format"] != "cedr-training-v1"
        or bundle["contract"] != contract
        or bundle["world_size"] != world
    ):
        raise ValueError(
            "resume requires the same training configuration and GPU topology"
        )
    for k, m in models.items():
        if m is not None:
            m.load_state_dict(bundle["models"][k], strict=True)
    for k, e in emas.items():
        e.load_state_dict(bundle["emas"][k])
    for k, oo in optimizers.items():
        for o, s in zip(oo, bundle["optimizers"][k], strict=True):
            o.load_state_dict(s)
    cursor.load_state_dict(bundle["cursor"])
    if balance is not None:
        if bundle["balance"] is None:
            raise ValueError("missing gradient-balancing state")
        balance.load_state_dict(bundle["balance"])
    set_rng(bundle["rank_rng"][rank])
    return bundle["update"]


def export_checkpoint(source, output, elf_selector=None, prompt_selector=None):
    """Export raw weights and EMAs, or an independently selected model pair."""
    bundle = load_tensor_file(source)
    if bundle.get("format") not in {"cedr-training-v1", "cedr-export-v1"}:
        raise ValueError("expected an CEDR training checkpoint or weight export")
    if elf_selector is None and prompt_selector is None:
        result = {k: bundle[k] for k in ("config", "models", "emas")}
    else:
        models = {}
        for name, selector in [("elf", elf_selector), ("prompt", prompt_selector)]:
            if name in bundle["models"]:
                if selector is None:
                    raise ValueError("select ELF and prompt independently")
                models[name] = selected_state(bundle, name, selector)
        result = dict(
            config=bundle["config"],
            models=models,
            emas={},
            selection=dict(elf=elf_selector, prompt=prompt_selector),
        )
    atomic_save(
        dict(
            format="cedr-export-v1",
            source_sha256=sha256(source),
            stage=bundle.get("stage", bundle.get("contract", {}).get("stage")),
            **result,
        ),
        output,
    )
