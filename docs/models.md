# Model packages

The six selected-weight packages are hosted at [xc91/CEDR](https://huggingface.co/xc91/CEDR). Each contains one independently selected model/prompt pair, a configuration, and SHA-256 hashes.

| Model ID | Stage | Model EMA | Prompt EMA | Steps | SCCFG |
|---|---|---|---|---|---:|
| `gsm8k-b-pre-nft` | Joint epoch 18 | .9999 | .9999 | 64 | 2 |
| `gsm8k-b-post-nft` | NFT update 300 | .99 | .99 | 64 | 2 |
| `math-l-pre-nft` | Joint epoch 21 | .9999 | .9999 | 64 | 3 |
| `math-l-post-nft` | NFT update 600 | .99 | .99 | 64 | 3 |
| `oci-l-pre-nft` | Joint epoch 13 | .9999 | .9999 | 128 | 3 |
| `oci-l-post-nft` | NFT update 100 | .9 | .9 | 128 | 3 |

The GSM8K-B post-NFT preset caps the decoded response at 767 tokens; the other presets use the available canvas after each prompt.

All presets use asynchronous clocks `(2.5, 2, 1.5)`, CFG 1, batch size 32, and a 1,024-token canvas. An ODE step is one denoiser call; terminal decoding adds one model call.

Learned weights use FP32 safetensors. Packages share the BF16 frozen token embedding table and tokenizer from `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The Qwen Transformer, projectors, and training caches are unnecessary for inference. Keep the selected model and prompt encoder together: their latent coordinates are matched.

```bash
cedr download --model gsm8k-b-pre-nft --output artifacts/gsm8k-b-pre
```

Only the requested model and shared inference assets are downloaded. The downloader resolves an immutable Hub commit and verifies file sizes and hashes. `download.json` records the resolved revision; pass it as `--revision COMMIT` to reproduce the download.

The validated release revision is `ce4e9cf93f173fe5e9a5c32e359a23ee27b88115`. Add `--revision ce4e9cf93f173fe5e9a5c32e359a23ee27b88115` to pin these packages.

The model packages are licensed under Apache 2.0 and retain Qwen attribution. The GitHub code is MIT licensed. See [evaluation.md](evaluation.md) for benchmark commands and numerical settings, and [model_validation.md](model_validation.md) for seed-42 release checks.
