# Inference-package validation

Seed-42 validation uses DMD2-style re-noising with eta 1, batch size 32, async `(2.5, 2, 1.5)`, and each package's default steps and SCCFG. These single-seed checks are separate from the paper's multi-seed estimates.

All 7,360 generated predictions and per-problem correctness decisions match the corresponding reference evaluations. Export checks verify exact selected tensors, matched representations, and strict architecture loading. Fixed-input GPU checks reproduce prompt features, denoiser outputs, sampled endpoints, and decoded tokens exactly for all eight models under ODE, EDM-style SDE, and DMD2-style sampling, including long prompts and partial batches. Full generation also verifies interrupted/resumed output equality.

| Model | Benchmark | Correct / total | Pass@1 | Plus correct / total | Plus pass@1 |
|---|---|---:|---:|---:|---:|
| gsm8k-b-pre-nft | gsm8k | 637/1319 | 48.29% | — | — |
| gsm8k-b-post-nft | gsm8k | 626/1319 | 47.46% | — | — |
| gsm8k-l-pre-nft | gsm8k | 879/1319 | 66.64% | — | — |
| gsm8k-l-post-nft | gsm8k | 898/1319 | 68.08% | — | — |
| math-l-pre-nft | math500 | 145/500 | 29.00% | — | — |
| math-l-post-nft | math500 | 158/500 | 31.60% | — | — |
| oci-l-pre-nft | humaneval | 57/164 | 34.76% | 54/164 | 32.93% |
| oci-l-pre-nft | mbpp | 123/378 | 32.54% | 101/378 | 26.72% |
| oci-l-post-nft | humaneval | 70/164 | 42.68% | 66/164 | 40.24% |
| oci-l-post-nft | mbpp | 110/378 | 29.10% | 88/378 | 23.28% |

Coding uses pinned official EvalPlus base/plus scoring without function-name repair. Optional alias scoring remains separately selectable. GSM8K uses canonical numerical scoring; MATH500 uses Math-Verify 0.9.0.

A clean noneditable installation passes 53 CPU tests covering package versions, selective downloading, file integrity, all three samplers, exact resumption, benchmark coverage, optimization, and epoch-tail rules. The eight-package release catalog is exercised through the installed downloader with local file transport; every selected file passes size and SHA-256 checks. Public downloads perform the same verification against an immutable Hub revision.

Two-H200 fine-tuning checks cover both selected states of GSM8K-L and OCI-L with effective batch 512 (2 × 32 × accumulation 8), three updates per state, both objective branches, gradients into both models, frozen embeddings/projectors, independent EMAs, fresh optimizers, task-specific learning rates, and save/reload continuation. These are short correctness checks, not convergence experiments. Fixed-panel cached/live features and losses match exactly at matched extraction chunk sizes; alternate chunks are checked with numerical tolerance.

Numerical validation uses H200 GPUs, PyTorch 2.11.0/CUDA 12.8, and Transformers 4.57.6. Different hardware, batch geometry, or numerical settings can change seeded outputs.
