# Inference-package validation

Validation uses seed 42, batch size 32, async `(2.5, 2, 1.5)`, and each package's default steps and SCCFG. These single-seed checks are separate from the paper's eight-seed estimates.

All 4,722 generated predictions and per-problem correctness decisions match the benchmark reference evaluations. Checks also cover exact selected-tensor export, strict loading, prompt features, denoiser outputs, sampled endpoints, vocabulary decoding, long prompts, partial batches, and interrupted/resumed generation.

| Model | Benchmark | Correct / total | Pass@1 | Plus correct / total | Plus pass@1 |
|---|---|---:|---:|---:|---:|
| gsm8k-b-pre-nft | gsm8k | 523/1319 | 39.65% | — | — |
| gsm8k-b-post-nft | gsm8k | 554/1319 | 42.00% | — | — |
| math-l-pre-nft | math500 | 103/500 | 20.60% | — | — |
| math-l-post-nft | math500 | 128/500 | 25.60% | — | — |
| oci-l-pre-nft | humaneval | 49/164 | 29.88% | 46/164 | 28.05% |
| oci-l-pre-nft | mbpp | 70/378 | 18.52% | 61/378 | 16.14% |
| oci-l-post-nft | humaneval | 54/164 | 32.93% | 53/164 | 32.32% |
| oci-l-post-nft | mbpp | 92/378 | 24.34% | 80/378 | 21.16% |

The installed public CLI was also checked from outside the source checkout, using only an inference package and prepared benchmark prompts. The coding sanitizer and base/plus decisions match the pinned official EvalPlus scorer without function-name repair. The CPU suite passes 33 tests, including package integrity, selective revision-pinned downloading, benchmark-cohort integrity, resumption, and optional function-name repair.

Standard and optional alias extraction each reproduce the corresponding saved programs for all 8,672 coding completions across the two models and eight seeds. The alias rule changes 2,432 programs; standard extraction remains the default. These extraction checks are separate from the single-seed accuracy table above.

All six packages were downloaded from [the model repository](https://huggingface.co/xc91/CEDR) and passed file-size and SHA-256 verification. The installed download command was also checked with implicit authentication disabled.

Numerical validation used H200 GPUs, PyTorch 2.11.0 with CUDA 12.8, and Transformers 4.57.6. Different hardware or numerical settings may change seeded outputs.
