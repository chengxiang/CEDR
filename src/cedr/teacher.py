"""Frozen causal Qwen, with differentiable suffix interventions for projectors."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import torch
from safetensors import safe_open
from .common import preserve_rng, autocast, artifact_identity


class _StopPrefix(Exception):
    pass


def load_embedding(path, tensor_name="model.embed_tokens.weight"):
    path = Path(path)
    if path.is_dir():
        index = json.loads((path / "model.safetensors.index.json").read_text())
        path = path / index["weight_map"][tensor_name]
    with safe_open(str(path), framework="pt", device="cpu") as f:
        key = tensor_name if tensor_name in f.keys() else "embedding"
        return f.get_tensor(key).detach().to(torch.bfloat16)


class QwenTeacher:
    def __init__(self, model, layers, device, *, revision=None, chunk_rows=32):
        from transformers import AutoModelForCausalLM

        self.device = torch.device(device)
        self.layers = tuple(map(int, layers))
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows < 1:
            raise ValueError("Qwen chunk size must be positive")
        with preserve_rng():
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model,
                    revision=revision,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
        if max(self.layers) > len(self.model.model.layers) or min(self.layers) < 1:
            raise ValueError("invalid post-block layer")
        self.identity = dict(
            model=str(model),
            revision=getattr(self.model.config, "_commit_hash", None) or revision,
            layers=list(self.layers),
            dtype="bfloat16",
            attention="sdpa_causal",
            artifact=artifact_identity(model) if Path(model).is_dir() else None,
        )

    @property
    def embedding(self):
        return self.model.get_input_embeddings().weight.detach()

    def base(self, batch):
        return self.model.model(
            input_ids=batch["ids"],
            attention_mask=batch["valid"].long(),
            use_cache=False,
        ).last_hidden_state

    def capture(self, batch, *, final=False):
        """Return original post-block activations. No teacher sampling or RNG consumption."""
        if len(batch["ids"]) > self.chunk_rows:
            parts = [
                self.capture(
                    {k: batch[k][lo : lo + self.chunk_rows] for k in ("ids", "valid")},
                    final=final,
                )
                for lo in range(0, len(batch["ids"]), self.chunk_rows)
            ]
            return (
                {
                    layer: torch.cat([p[0][layer] for p in parts])
                    for layer in self.layers
                },
                torch.cat([p[1] for p in parts]) if final else None,
            )
        captured = {}
        handles = []
        for layer in self.layers:

            def hook(module, args, out, layer=layer):
                captured[layer] = (out[0] if isinstance(out, tuple) else out).detach()
                if layer == max(self.layers) and not final:
                    raise _StopPrefix()

            handles.append(
                self.model.model.layers[layer - 1].register_forward_hook(hook)
            )
        try:
            with preserve_rng(), torch.no_grad(), autocast(self.device):
                try:
                    last = self.base(batch)
                except _StopPrefix:
                    last = None
        finally:
            for h in handles:
                h.remove()
        return captured, last

    def intervene(self, batch, layer, replacement, mask):
        """Frozen parameters, enabled autograd: the suffix differentiates into replacement."""
        if len(batch["ids"]) > self.chunk_rows:
            return torch.cat(
                [
                    self.intervene(
                        {
                            k: batch[k][lo : lo + self.chunk_rows]
                            for k in ("ids", "valid")
                        },
                        layer,
                        replacement[lo : lo + self.chunk_rows],
                        mask[lo : lo + self.chunk_rows],
                    )
                    for lo in range(0, len(batch["ids"]), self.chunk_rows)
                ]
            )

        def hook(module, args, out):
            hidden = out[0] if isinstance(out, tuple) else out
            changed = torch.where(
                mask.unsqueeze(-1), replacement.to(hidden.dtype), hidden
            )
            return (changed, *out[1:]) if isinstance(out, tuple) else changed

        handle = self.model.model.layers[layer - 1].register_forward_hook(hook)
        try:
            with autocast(self.device):
                return self.base(batch)
        finally:
            handle.remove()

    def logits(self, hidden):
        # Full-vocabulary probabilities in FP32; callers chunk the token axis.
        with torch.autocast(self.device.type, enabled=False):
            head = self.model.get_output_embeddings()
            return torch.nn.functional.linear(
                hidden.float(),
                head.weight.float(),
                None if head.bias is None else head.bias.float(),
            )
