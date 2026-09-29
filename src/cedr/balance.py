"""Opt-in prompt-only accumulated DP-mean gradient balancing; no ELF changes."""

import math
import torch
import torch.distributed as dist


def row_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("prompt target/mask shape mismatch")
    lengths = mask.sum(1)
    if not bool((lengths > 0).all()) or target.requires_grad:
        raise ValueError("MSE needs nonempty real prompts and detached teacher targets")
    squared = (prediction.float() - target.float()).square().mean(-1)
    return ((squared * mask).sum(1) / lengths).mean()


class PromptMSEBalance:
    def __init__(self, decay=0.99, epsilon=1e-12, target_ratio=1.0):
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.target_ratio = float(target_ratio)
        if not math.isfinite(self.target_ratio) or self.target_ratio <= 0:
            raise ValueError("MSE target ratio must be finite and positive")
        self.a = None
        self.b = None
        self.updates = 0
        self.buffers = None
        self.microsteps = 0

    def state_dict(self):
        if self.microsteps or self.buffers is not None:
            raise ValueError("balanced MSE checkpoints require an optimizer boundary")
        return dict(
            version=1,
            decay=self.decay,
            epsilon=self.epsilon,
            target_ratio=self.target_ratio,
            a=self.a,
            b=self.b,
            updates=self.updates,
        )

    def load_state_dict(self, value):
        if (
            value["version"] != 1
            or value["decay"] != self.decay
            or value["epsilon"] != self.epsilon
            or value["target_ratio"] != self.target_ratio
        ):
            raise ValueError("gradient balance contract changed")
        self.a = value["a"]
        self.b = value["b"]
        self.updates = int(value["updates"])
        if self.updates < 0 or (
            (self.a is None or self.b is None) != (self.updates == 0)
        ):
            raise ValueError("invalid norm EMA state")
        if self.updates and not all(
            math.isfinite(x) and x >= 0 for x in (self.a, self.b)
        ):
            raise ValueError("nonfinite norm EMA state")

    def accumulate(self, loss, parameters, accumulation):
        gradients = torch.autograd.grad(
            loss / accumulation, parameters, allow_unused=False
        )
        if self.buffers is None:
            self.buffers = [g.detach().float().clone() for g in gradients]
        else:
            for b, g in zip(self.buffers, gradients):
                b.add_(g.detach())
        self.microsteps += 1

    def combine(self, parameters, accumulation, clip_norm=1.0):
        if self.microsteps != accumulation:
            raise ValueError("incomplete auxiliary accumulation")
        if any(p.grad is None for p in parameters):
            raise ValueError("missing downstream prompt gradient")
        downstream = [p.grad for p in parameters]
        aux = self.buffers
        # Prompt is deliberately outside DDP. Average each component exactly once.
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            # Coalesced flat buckets bound memory and collective-launch overhead.
            for values in (downstream, aux):
                bucket = []
                size = 0

                def reduce_bucket(items):
                    flat = torch.cat([x.reshape(-1) for x in items])
                    dist.all_reduce(flat)
                    flat.div_(world)
                    offset = 0
                    for x in items:
                        x.copy_(flat[offset : offset + x.numel()].view_as(x))
                        offset += x.numel()

                for value in values:
                    bucket.append(value)
                    size += value.numel()
                    if size >= 8 * 1024 * 1024:
                        reduce_bucket(bucket)
                        bucket = []
                        size = 0
                if bucket:
                    reduce_bucket(bucket)
        dn = torch.linalg.vector_norm(
            torch.stack([x.float().norm() for x in downstream])
        )
        mn = torch.linalg.vector_norm(torch.stack([x.norm() for x in aux]))
        dot = sum((d.float() * m).sum() for d, m in zip(downstream, aux))
        d, m = float(dn), float(mn)
        if not all(math.isfinite(x) for x in (d, m, float(dot))):
            raise FloatingPointError("nonfinite component gradient")
        self.a = d if self.updates == 0 else self.decay * self.a + (1 - self.decay) * d
        self.b = m if self.updates == 0 else self.decay * self.b + (1 - self.decay) * m
        coefficient = self.target_ratio * self.a / max(self.b, self.epsilon)
        if not math.isfinite(coefficient):
            raise FloatingPointError("nonfinite MSE coefficient")
        for p, g in zip(parameters, aux):
            p.grad.add_(g, alpha=coefficient)
        combined = torch.linalg.vector_norm(
            torch.stack([p.grad.float().norm() for p in parameters])
        )
        if not bool(torch.isfinite(combined)):
            raise FloatingPointError("nonfinite combined gradient")
        self.updates += 1
        self.buffers = None
        self.microsteps = 0
        values = dict(
            prompt_mse_lambda=coefficient,
            prompt_downstream_norm=d,
            prompt_mse_norm=m,
            prompt_mse_target_ratio=self.target_ratio,
            prompt_downstream_norm_ema=self.a,
            prompt_mse_norm_ema=self.b,
            prompt_mse_achieved_ratio=coefficient * m / max(d, self.epsilon),
            prompt_mse_cosine=float(dot) / max(d * m, self.epsilon),
            prompt_combined_norm=float(combined),
            prompt_clip_scale=min(1.0, clip_norm / (float(combined) + 1e-6)),
            prompt_mse_balance_updates=self.updates,
        )
        return {
            k: torch.tensor(v, device=dn.device, dtype=torch.float32)
            for k, v in values.items()
        }
