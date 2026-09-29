from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    BottleneckTextProj,
    DEFAULT_BIAS_INIT,
    DEFAULT_KERNEL_INIT,
    RMSNorm,
    TextRotaryEmbeddingFast,
    _make_linear,
)

from .model import ELFBlock


@dataclass(frozen=True)
class FrozenQwenEmbeddingPromptEncoderConfig:
    """Prompt encoder architecture with frozen Qwen token embeddings.

    The external embedding table is deliberately not a parameter or a
    persistent buffer.  A checkpoint therefore contains the trainable bridge
    and Transformer body only and must be paired with the pinned
    embedding table when it is restored.
    """

    vocab_size: int = 151_936
    max_length: int = 1024
    external_embedding_dim: int = 2_560
    bottleneck_dim: int = 512
    hidden_size: int = 1_280
    depth: int = 6
    num_heads: int = 16
    mlp_ratio: float = 4.0
    output_dim: int = 1_024
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    @classmethod
    def from_dict(
        cls, values: dict[str, Any] | None
    ) -> "FrozenQwenEmbeddingPromptEncoderConfig":
        values = dict(values or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown frozen-Qwen prompt-encoder config fields: {unknown}"
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrozenQwenEmbeddingPromptEncoder(nn.Module):
    """Prompt encoder with a frozen external table and trainable bottleneck.

    The frontend is exactly ``2560 -> 512 -> 1280`` for the CEDR-L configuration:
    both operations are linear, the first has no bias, the second has a bias,
    and there is no activation between them.  ``external_token_embedding`` is
    registered with ``persistent=False`` so it follows device/dtype moves but
    is absent from parameters, optimizer groups, EMAs, and checkpoints.
    """

    def __init__(
        self,
        config: FrozenQwenEmbeddingPromptEncoderConfig,
        external_token_embedding: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        cfg = config
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if cfg.max_length <= 0 or cfg.depth <= 0:
            raise ValueError("max_length and depth must be positive")
        if cfg.external_embedding_dim <= 0 or cfg.bottleneck_dim <= 0:
            raise ValueError(
                "external_embedding_dim and bottleneck_dim must be positive"
            )
        if tuple(external_token_embedding.shape) != (
            cfg.vocab_size,
            cfg.external_embedding_dim,
        ):
            raise ValueError(
                "external token embedding shape mismatch: got "
                f"{tuple(external_token_embedding.shape)}, expected "
                f"{(cfg.vocab_size, cfg.external_embedding_dim)}"
            )
        if not external_token_embedding.is_floating_point():
            raise ValueError("external token embedding must be floating point")

        embedding = external_token_embedding.detach().contiguous()
        embedding.requires_grad_(False)
        self.register_buffer("external_token_embedding", embedding, persistent=False)
        self.input_projection = BottleneckTextProj(
            cfg.external_embedding_dim,
            cfg.hidden_size,
            cfg.bottleneck_dim,
        )
        self.rope = TextRotaryEmbeddingFast(
            dim=cfg.hidden_size // cfg.num_heads,
            pt_seq_len=cfg.max_length,
            num_empty_token=0,
        )
        self.blocks = nn.ModuleList(
            [
                ELFBlock(
                    cfg.hidden_size,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attn_drop=cfg.attn_dropout,
                    proj_drop=cfg.proj_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.output_norm = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.output_projection = _make_linear(
            cfg.hidden_size,
            cfg.output_dim,
            bias=True,
            kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or prompt_mask.ndim != 2:
            raise ValueError("input_ids and prompt_mask must both be [B,L]")
        if input_ids.shape != prompt_mask.shape:
            raise ValueError("input_ids and prompt_mask shapes must match")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError(
                f"prompt length {input_ids.shape[1]} exceeds configured "
                f"maximum {self.config.max_length}"
            )
        valid = prompt_mask.to(device=input_ids.device, dtype=torch.bool)
        if not torch.compiler.is_compiling():
            if input_ids.numel() and (
                int(input_ids.min()) < 0
                or int(input_ids.max()) >= self.config.vocab_size
            ):
                raise ValueError("prompt token id is outside the configured vocabulary")
            if not bool(valid.any(dim=1).all()):
                raise ValueError("every prompt must contain at least one valid token")
        embedded = F.embedding(input_ids, self.external_token_embedding)
        # Outside autocast, cast gathered embeddings to the projection
        # weights' dtype. The full embedding table stays BF16 and frozen.
        if (
            embedded.dtype != self.input_projection.proj1.weight.dtype
            and not torch.is_autocast_enabled(embedded.device.type)
        ):
            embedded = embedded.to(self.input_projection.proj1.weight.dtype)
        hidden = self.input_projection(embedded)
        rope = lambda tensor: self.rope(tensor, num_empty_token=0)
        for block in self.blocks:
            hidden = block(
                hidden,
                rope_fn=rope,
                attention_mask=valid,
                deterministic=deterministic,
            )
        output = self.output_projection(self.output_norm(hidden))
        return output * valid.unsqueeze(-1).to(output.dtype)

    @property
    def parameter_count(self) -> int:
        """Count checkpointed/trainable tensors; excludes the Qwen table."""
        return sum(parameter.numel() for parameter in self.parameters())
