"""CEDR diffusion Transformer."""

import copy

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import (
    Attention,
    BottleneckTextProj,
    FinalLayer,
    RMSNorm,
    SwiGLUFFN,
    TextRotaryEmbeddingFast,
    TimestepEmbedder,
    DEFAULT_KERNEL_INIT,
    DEFAULT_BIAS_INIT,
    NORMAL_INIT_002,
    _make_linear,
)


class ELFBlock(nn.Module):
    """Transformer block shared by the backbone and prompt encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out = self.attn(
            x_normed,
            rope_fn,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class CEDR(nn.Module):
    """Latent diffusion Transformer with a factored vocabulary decoder."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        decoder_dim: int = 0,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
        feature_group_time_conditioning: bool = False,
        feature_group_time_names: tuple[str, ...] = (),
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.decoder_dim = int(decoder_dim or text_encoder_dim)
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.feature_group_time_conditioning = bool(feature_group_time_conditioning)
        self.feature_group_time_names = tuple(
            str(name) for name in feature_group_time_names
        )
        if self.feature_group_time_conditioning:
            if len(self.feature_group_time_names) < 2:
                raise ValueError(
                    "feature-group time conditioning requires at least two names"
                )
            if len(set(self.feature_group_time_names)) != len(
                self.feature_group_time_names
            ):
                raise ValueError("feature-group time names must be unique")

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(
            2 * text_encoder_dim, text_encoder_dim, bias=True
        )

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(
            text_encoder_dim, hidden_size, bottleneck_dim
        )

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError(
                "num_time_tokens must be positive for prefix time conditioning"
            )
        if self.feature_group_time_conditioning:
            first = TimestepEmbedder(hidden_size)
            self.t_embedders = nn.ModuleDict(
                {
                    name: first if index == 0 else copy.deepcopy(first)
                    for index, name in enumerate(self.feature_group_time_names)
                }
            )
        else:
            self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, hidden_size)
            )
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                torch.empty(1, num_model_mode_tokens, hidden_size)
            )
            NORMAL_INIT_002(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim,
            pt_seq_len=max_length,
            num_empty_token=prefix_total,
        )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(
                ELFBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if in_drop_range else 0.0,
                    proj_drop=proj_drop if in_drop_range else 0.0,
                )
            )

        # Final flow-matching output head.
        self.final_layer = FinalLayer(
            hidden_size, patch_size=1, out_channels=text_encoder_dim
        )

        # Factored decoder unembedding: hidden -> decoder_dim -> vocab.
        bn = self.decoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def build_context(self, t, self_cond_cfg_scale=None):
        if (
            not isinstance(t, (tuple, list))
            and not self.feature_group_time_conditioning
        ):
            time = self.t_embedder(t)
            batch = t.shape[0]
        else:
            if not isinstance(t, (tuple, list)):
                t = tuple(t for _ in self.feature_group_time_names)
            if not t or any(value.shape != t[0].shape for value in t[1:]):
                raise ValueError(
                    "local times must be nonempty and have matching shapes"
                )
            embeddings = (
                [
                    self.t_embedders[name](x)
                    for name, x in zip(self.feature_group_time_names, t, strict=True)
                ]
                if self.feature_group_time_conditioning
                else [self.t_embedder(x) for x in t]
            )
            # FP64 averaging preserves identical-clock embeddings exactly.
            time = torch.stack(embeddings).double().mean(0).float()
            batch = t[0].shape[0]
        prefix = [self.t_emb_tokens.expand(batch, -1, -1) + time[:, None]]
        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            prefix.append(
                self.self_cond_cfg_tokens.expand(batch, -1, -1)
                + self.self_cond_cfg_embedder(self_cond_cfg_scale)[:, None]
            )
        return prefix

    def forward(
        self,
        x: torch.Tensor,
        t,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        decoder_return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (N,S,C)/(N,S,2C); t: scalar or an enabled async clock tuple.

        When ``decoder_return_features`` is true, the second result is the
        post-GELU decoder feature tensor rather than full-vocabulary logits.
        This lets training compute exact selected-token CE in bounded chunks.
        """
        B = x.shape[0]

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast("cuda", enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            x = self.text_proj(x.float())
            context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        # Prepend learnable model-mode tokens (gated by decoder_step_active).
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif (
                isinstance(decoder_step_active, torch.Tensor)
                and decoder_step_active.dim() > 0
            ):
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(
                    (B, self.num_model_mode_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    (B, prefix_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        actual_empty_tokens = prefix_len + model_mode_offset

        def rope_fn(tensor: torch.Tensor) -> torch.Tensor:
            return self.feat_rope(tensor, num_empty_token=actual_empty_tokens)

        use_checkpoint = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        )
        for block in self.blocks:
            if use_checkpoint:

                def _block_forward(
                    hidden: torch.Tensor, block: ELFBlock = block
                ) -> torch.Tensor:
                    return block(
                        hidden,
                        rope_fn=rope_fn,
                        attention_mask=attention_mask,
                        deterministic=deterministic,
                    )

                x = checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(
                    x,
                    rope_fn=rope_fn,
                    attention_mask=attention_mask,
                    deterministic=deterministic,
                )

        x = x[:, prefix_len + model_mode_offset :]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        with torch.amp.autocast("cuda", enabled=False):
            decoder_logits = None
            if decoder_step_active is not None:
                x_f32 = x.float()
                hidden = F.gelu(
                    x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh"
                )
                decoder_logits = (
                    hidden
                    if decoder_return_features
                    else hidden @ self.unembed_kernel + self.unembed_bias
                )
            output = self.final_layer(x.float())
        return output, decoder_logits
