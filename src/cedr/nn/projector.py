"""Differentiable covariance whitening and reconstruction bottleneck."""
from __future__ import annotations
from typing import Any, Mapping
import torch
from torch import nn

def canonical_thin_qr(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Thin QR with a deterministic nonnegative diagonal in ``R``."""

    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("QR source must have shape [hidden,bottleneck]")
    q, r = torch.linalg.qr(matrix.float(), mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return (q * signs.unsqueeze(0)).contiguous(), (signs.unsqueeze(1) * r).contiguous()


class DiffQRReconstructionBottleneck(nn.Module):
    """One FP32 covariance-aware encoder and FP32 reconstruction decoder.

    With fixed ``W = covariance^-1/2`` and ``S = covariance^1/2``, the
    effective encoder and initialization are

    ``Q = canonical_thin_qr(R)``, ``z = (h-mu) W Q``, ``R0 = S P``, and
    ``D0 = S Q0``.  ``P`` is the freshly computed PCA effective encoder.
    Both ``R`` and ``D`` are trainable FP32 parameters.
    """

    def __init__(
        self,
        *,
        layer: int,
        mean: torch.Tensor,
        hidden_whitener: torch.Tensor,
        hidden_sqrt: torch.Tensor,
        pca_effective_encoder: torch.Tensor,
        inverse_atol: float = 2e-3,
        conversion_rtol: float = 1e-5,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.layer = int(layer)
        if self.layer < 1:
            raise ValueError(f"unsupported bottleneck layer {self.layer}")
        mean = mean.detach().float().contiguous()
        whitener = hidden_whitener.detach().float().contiguous()
        sqrt = hidden_sqrt.detach().float().contiguous()
        pca = pca_effective_encoder.detach().float().contiguous()
        hidden = int(mean.numel())
        expected_dimension = pca.shape[1]
        if mean.ndim != 1:
            raise ValueError("mean must be a vector")
        if whitener.shape != (hidden, hidden) or sqrt.shape != (hidden, hidden):
            raise ValueError("whitener and covariance sqrt must be square hidden matrices")
        if pca.shape != (hidden, expected_dimension):
            raise ValueError(
                f"layer {self.layer} PCA encoder must have shape "
                f"({hidden}, {expected_dimension})"
            )
        if not all(torch.isfinite(value).all() for value in (mean, whitener, sqrt, pca)):
            raise ValueError("bottleneck geometry contains nonfinite values")
        identity_error = float(
            (
                whitener.double() @ sqrt.double()
                - torch.eye(hidden, dtype=torch.float64)
            )
            .abs()
            .max()
        )
        if identity_error > float(inverse_atol):
            raise ValueError(
                f"whitener/sqrt inverse error {identity_error:.6g} exceeds {inverse_atol}"
            )
        raw = (sqrt @ pca).contiguous()
        q0, r0 = canonical_thin_qr(raw)
        if float(torch.diagonal(r0).abs().min()) <= 0.0:
            raise ValueError("fresh PCA initialization is rank deficient")
        effective0 = whitener @ q0
        conversion_error = float(
            (effective0 - pca).norm() / pca.norm().clamp_min(1e-20)
        )
        if conversion_error > float(conversion_rtol):
            raise ValueError(
                f"PCA-to-DiffQR conversion error {conversion_error:.6g} exceeds "
                f"{conversion_rtol}"
            )
        decoder = (sqrt @ q0).contiguous()

        self.hidden_size = hidden
        self.dimension = expected_dimension
        self.inverse_identity_max_abs = identity_error
        self.conversion_relative_error = conversion_error
        self.source_provenance = dict(provenance or {})
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("hidden_whitener", whitener, persistent=False)
        self.register_buffer("hidden_sqrt", sqrt, persistent=False)
        self.register_buffer("initial_q", q0.detach().clone(), persistent=False)
        self.raw_parameter = nn.Parameter(raw)
        self.decoder_layer = nn.Linear(
            expected_dimension, hidden, bias=False, dtype=torch.float32
        )
        with torch.no_grad():
            self.decoder_layer.weight.copy_(decoder)

    @property
    def decoder(self) -> nn.Parameter:
        """The FP32 ``D[hidden,d]`` matrix, stored with nn.Linear semantics."""

        return self.decoder_layer.weight

    def canonical_qr(
        self, raw_parameter: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_parameter if raw_parameter is None else raw_parameter
        if raw.dtype != torch.float32 or raw.shape != self.raw_parameter.shape:
            raise ValueError("DiffQR raw parameter must retain its FP32 production shape")
        return canonical_thin_qr(raw)

    def effective_encoder(self, raw_parameter: torch.Tensor | None = None) -> torch.Tensor:
        q, _ = self.canonical_qr(raw_parameter)
        return self.hidden_whitener @ q

    def encode(
        self, hidden: torch.Tensor, *, raw_parameter: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden state width does not match bottleneck geometry")
        return (hidden.float() - self.mean) @ self.effective_encoder(raw_parameter)

    def reconstruct(
        self,
        hidden: torch.Tensor,
        *,
        raw_parameter: torch.Tensor | None = None,
        decoder: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected_decoder = self.decoder if decoder is None else decoder
        if selected_decoder.dtype != torch.float32 or selected_decoder.shape != self.decoder.shape:
            raise ValueError("decoder override must retain its FP32 production shape")
        latent = self.encode(hidden, raw_parameter=raw_parameter)
        return self.mean + latent @ selected_decoder.T

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(hidden)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, torch.Tensor]:
        q, r = self.canonical_qr()
        identity = torch.eye(self.dimension, device=q.device, dtype=torch.float64)
        covariance_coordinates = self.hidden_sqrt.double() @ (
            self.hidden_whitener.double() @ q.double()
        )
        singular_values = torch.linalg.svdvals(q.double().T @ self.initial_q.double())
        singular_values = singular_values.clamp(0.0, 1.0)
        return {
            "qr_orthogonality_max_abs": (q.double().T @ q.double() - identity).abs().max(),
            "covariance_identity_max_abs": (
                covariance_coordinates.T @ covariance_coordinates - identity
            ).abs().max(),
            "qr_diagonal_min_abs": torch.diagonal(r).abs().min(),
            "pca_subspace_mean_cos2": singular_values.square().mean(),
            "raw_norm": self.raw_parameter.norm(),
            "decoder_norm": self.decoder.norm(),
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "dimension": self.dimension,
            "hidden_size": self.hidden_size,
            "initialization": "fresh_pca_covariance_aware",
            "parameterization": "differentiable_canonical_thin_qr",
            "trainable_parameters": ["raw_parameter", "decoder_layer.weight"],
            "parameter_dtype": "float32",
            "inverse_identity_max_abs": self.inverse_identity_max_abs,
            "conversion_relative_error": self.conversion_relative_error,
            **self.source_provenance,
        }
