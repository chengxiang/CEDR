# Model packages

CEDR 0.2 or later is required for the sampler-aware `cedr-inference-v2` packages. The loader also supports `cedr-inference-v1` packages. The format version prevents older code from silently ignoring the stochastic sampler preset.

The eight selected-weight packages are hosted at [xc91/CEDR](https://huggingface.co/xc91/CEDR). Each contains one independently selected model/prompt pair, a configuration, and SHA-256 hashes.

| Model ID | Stage | Model EMA | Prompt EMA | Steps | SCCFG |
|---|---|---|---|---|---:|
| `gsm8k-b-pre-nft` | Joint epoch 18 | .9999 | .9999 | 64 | 2 |
| `gsm8k-b-post-nft` | NFT update 300 | .99 | .99 | 64 | 2 |
| `gsm8k-l-pre-nft` | Joint epoch 21 | .9999 | .9999 | 64 | 2 |
| `gsm8k-l-post-nft` | NFT update 600 | .99 | .99 | 64 | 2 |
| `math-l-pre-nft` | Joint epoch 21 | .9999 | .9999 | 64 | 3 |
| `math-l-post-nft` | NFT update 600 | .99 | .99 | 64 | 3 |
| `oci-l-pre-nft` | Joint epoch 14 | .9999 | .9999 | 128 | 3 |
| `oci-l-post-nft` | NFT update 100 | .99 | .99 | 128 | 3 |

Both GSM8K post-NFT presets cap the decoded response at 767 tokens; the other presets use the available canvas after each prompt.

All presets use asynchronous clocks `(2.5, 2, 1.5)`, CFG 1, batch size 32, and a 1,024-token canvas. Defaults use DMD2-style re-noising with eta 1. Each step is one denoiser call at CFG 1; terminal decoding adds one model call. ODE and EDM-style SDE are selectable with `--sampler`; see [sampler conventions](samplers.md).

Learned weights use FP32 safetensors. Packages share the BF16 frozen token embedding table and tokenizer from `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The Qwen Transformer, projectors, and training caches are unnecessary for inference. Keep the selected model and prompt encoder together: their latent coordinates are matched.

```bash
cedr download --model gsm8k-b-pre-nft --output artifacts/gsm8k-b-pre
```

Only the requested model, matched representation, and shared inference assets are downloaded. The downloader resolves an immutable Hub commit and verifies file sizes and hashes. `download.json` records the resolved revision; pass it as `--revision COMMIT` to reproduce the download.

The model packages are licensed under Apache 2.0 and retain Qwen attribution. The GitHub code is MIT licensed. See [evaluation.md](evaluation.md) for benchmark commands and numerical settings, and [model_validation.md](model_validation.md) for seed-42 release checks.

## Matched representations

Each current download includes `representation.pt`, shared by the pre-NFT and post-NFT model of its family. It contains the layer means and FP32 effective whitened encoders for layers 16/24/32 (256/256/512 dimensions), plus reconstruction matrices. GSM8K-B and GSM8K-L share the same learned representation tensors. Generation does not use this artifact. Joint fine-tuning and feature-cache construction use it to preserve the model's latent coordinates; no additional covariance fitting is needed. See [fine-tuning](finetuning.md).
