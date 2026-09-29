"""Exact selected-token CE for the factored vocabulary decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_factored_decoder_ce(
    decoder_features: torch.Tensor,
    targets: torch.Tensor,
    selection_mask: torch.Tensor,
    unembed_kernel: torch.Tensor,
    unembed_bias: torch.Tensor,
    *,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact CE sum plus detached per-token CE and predictions.

    Only selected response positions instantiate vocabulary logits.  The CE is
    otherwise identical to ``cross_entropy(features @ W + b, target)`` in
    FP32, including its gradient with respect to features, ``W``, and ``b``.
    Non-selected canvas entries are zero CE and prediction ``-1``.
    """

    if decoder_features.ndim != 3:
        raise ValueError("decoder features must have shape [batch,sequence,dimension]")
    if targets.shape != decoder_features.shape[:2] or selection_mask.shape != targets.shape:
        raise ValueError("decoder targets/mask do not match the feature token axes")
    if unembed_kernel.ndim != 2 or unembed_kernel.shape[0] != decoder_features.shape[-1]:
        raise ValueError("unembedding kernel does not match the decoder feature dimension")
    if unembed_bias.shape != (unembed_kernel.shape[1],):
        raise ValueError("unembedding bias does not match the vocabulary dimension")
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    flat_mask = selection_mask.reshape(-1).bool()
    selected_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    ce_canvas = torch.zeros(targets.numel(), device=decoder_features.device, dtype=torch.float32)
    prediction_canvas = torch.full(
        (targets.numel(),), -1, device=decoder_features.device, dtype=torch.long
    )
    if selected_indices.numel() == 0:
        # Keep a zero-valued dependency on every decoder parameter for ranks
        # whose Bernoulli draw contains no CE rows.
        zero = (
            decoder_features.sum() * 0.0
            + unembed_kernel.sum() * 0.0
            + unembed_bias.sum() * 0.0
        )
        return zero, ce_canvas.view_as(targets), prediction_canvas.view_as(targets)

    flat_features = decoder_features.reshape(-1, decoder_features.shape[-1])
    flat_targets = targets.reshape(-1)
    graph_sums = []
    with torch.amp.autocast("cuda", enabled=False):
        kernel = unembed_kernel.float()
        bias = unembed_bias.float()
        for begin in range(0, selected_indices.numel(), chunk_tokens):
            indices = selected_indices[begin : begin + chunk_tokens]
            features = flat_features.index_select(0, indices).float()
            labels = flat_targets.index_select(0, indices)
            logits = features @ kernel + bias
            losses = F.cross_entropy(logits, labels, reduction="none")
            graph_sums.append(losses.sum())
            ce_canvas.index_copy_(0, indices, losses.detach())
            prediction_canvas.index_copy_(0, indices, logits.detach().argmax(dim=-1))
    return (
        torch.stack(graph_sums).sum(),
        ce_canvas.view_as(targets),
        prediction_canvas.view_as(targets),
    )
