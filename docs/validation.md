# Testing

Install the test dependencies and run the CPU suite:

```bash
pip install -e '.[test]'
pytest -q
cedr validate --output /path/to/cpu-check
```

Tests cover covariance whitening, teacher suffix gradients, causal masks, terminal tokens, question-only conditioning, mixed losses, gradient balancing, independent EMAs, checkpoint export, and optimizer-boundary resume. Small training fixtures exercise the flow, prompt, joint, and NFT stages. Scoring checks verify answer extraction and reject reuse when prediction inputs change.

## GPU checks

Supply the pinned Qwen model and fitted representation parameters:

```bash
cedr validate --gpu --teacher /path/to/pinned-qwen \
  --representation /path/to/representation.pt --output /path/to/gpu-check
torchrun --standalone --nproc-per-node=2 tests/distributed_check.py \
  --output /path/to/distributed-check
```

Use a fresh output directory for each check. The GPU panel includes long prompts, cached/live feature and loss comparisons, different Qwen chunk sizes, compiled updates with a small backbone, and generation. The distributed check covers gradient averaging, loss normalization, replicated optimizer updates, and optimizer-state restoration.

Cached/live features and losses match exactly with matched extraction chunks. Different chunk sizes can introduce BF16 rounding differences, so cross-chunk comparisons use numerical tolerances. These small checks validate implementation behavior; they do not measure training convergence or benchmark accuracy.

Optional scoring dependencies are required for Math-Verify and EvalPlus; code execution also requires Linux user-namespace support and bubblewrap.
