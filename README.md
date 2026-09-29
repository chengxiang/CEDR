# CEDR: Continuous Embedding Diffusion for Reasoning

Continuous latent diffusion for reasoning with frozen Qwen representations and a learned prompt encoder.

Paper: [Reasoning with Continuous Latent Diffusion](https://arxiv.org/pdf/2609.35694).

The training pipeline consists of covariance-aware teacher-CE projector learning, flow/decoder training, prompt imitation with frozen Qwen embeddings, joint training, and NFT.

The implementation includes cached and live-Qwen features, independently selectable ELF/prompt EMAs, synchronous training clocks, asynchronous inference, and task-specific correctness rewards. OCI joint training enables the 2:1 gradient-balanced prompt-MSE term; math joint training and all NFT training disable it.

## Install

Use Python 3.10 or later and a CUDA-capable PyTorch installation. Tested with PyTorch 2.11.0/CUDA 12.8 and Transformers 4.57.6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[data,math,test]'
# Optional official EvalPlus scoring and coding rewards:
pip install -e '.[code]'
cedr --help
pytest -q
cedr validate --output /tmp/cedr-check
```

Code execution rewards require Linux **bubblewrap** (`bwrap`) with user namespaces enabled. Execution is isolated from the project tree and network. Sandbox startup errors stop scoring; they are not converted into incorrect answers. See [evaluation](docs/evaluation.md).

## Released models

[Six inference packages](https://huggingface.co/xc91/CEDR) cover GSM8K-B, MATH-L, and OCI-L, before and after NFT. Downloaded packages contain the matched denoiser, vocabulary decoder, prompt encoder, frozen embedding table, and tokenizer.

```bash
cedr download --model gsm8k-b-pre-nft --output artifacts/gsm8k-b-pre
cedr prepare-benchmark --benchmark gsm8k --checkpoint artifacts/gsm8k-b-pre \
  --output artifacts/gsm8k-test
cedr generate --checkpoint artifacts/gsm8k-b-pre --data artifacts/gsm8k-test \
  --seed 42 --output artifacts/gsm8k-seed42
cedr score --config artifacts/gsm8k-b-pre/config.json --data artifacts/gsm8k-test \
  --predictions artifacts/gsm8k-seed42/predictions.jsonl --workers 8 \
  --output artifacts/gsm8k-seed42/scoring
```

See [model selectors and presets](docs/models.md) and [evaluation commands](docs/evaluation.md). Model artifacts are Apache 2.0; this code is MIT.

Coding scoring defaults to standard EvalPlus extraction. Use `cedr score --function-name-repair` for the separately labeled single-function alias comparison; see the evaluation guide for its rule and output records.

## Configurations

| Configuration | Backbone | Flow epochs | Prompt MSE epochs | Joint epochs | NFT updates |
|---|---|---:|---:|---:|---:|
| `configs/gsm8k_b.yaml` | B | 12 | 60 | 6 (to 18) | 300 |
| `configs/gsm8k_l.yaml` | L | 12 | 30 | 9 (to 21) | 500 |
| `configs/math_l.yaml` | L | 12 | 10 | 9 (to 21) | 600 |
| `configs/oci_l.yaml` | L | 12 | 10 | 1 (to 13) | 100 |

Training durations, EMA selectors, and inference settings are configurable.

## Run the pipeline

The [pipeline guide](docs/pipeline.md) lists every command, data schema, eligibility rule, and artifact dependency. Supply/download the pinned Qwen snapshot and prepare normalized JSONL; training populations are prepared separately. Inference packages are available on Hugging Face.

Set paths to storage outside the repository:

```bash
recipe=configs/math_l.yaml
artifacts=/path/to/cedr-artifacts
teacher=/path/to/Qwen3-4B-Instruct-2507

cedr prepare --config "$recipe" --input "$artifacts/train.jsonl" \
  --tokenizer "$teacher" --output "$artifacts/data"
cedr covariance --config "$recipe" --data "$artifacts/projector-data" \
  --teacher "$teacher" --output "$artifacts/covariance.pt"
cedr projectors --config "$recipe" --data "$artifacts/projector-data" \
  --teacher "$teacher" --covariance "$artifacts/covariance.pt" \
  --output "$artifacts/projectors"
```

`projector-data` is a separately prepared representation-fitting population; see the guide for weighted sampling. Covariance can instead be estimated over its full eligible source population.

Cached features are the default. To avoid building a large cache, pass `--features live --teacher ... --representation ...` to any training stage. Live extraction is frozen/causal, preserves training RNG, projects in FP32, and returns the same BF16 target contract. Qwen forward chunk size is independent of the optimization batch.

```bash
# Example: live flow training on two GPUs (effective supervised batch stays 512).
torchrun --standalone --nproc-per-node=2 -m cedr train flow \
  --config "$recipe" --data "$artifacts/data" --features live \
  --teacher "$teacher" --representation "$artifacts/projectors/representation.pt" \
  --qwen-chunk 8 --micro-batch 32 --output "$artifacts/flow"
```

The remaining stages and cache workflow are in [pipeline.md](docs/pipeline.md). Slurm is optional; [examples/slurm](examples/slurm) contains a scheduler wrapper separate from the library.

## Testing

- Tests cover whitening, teacher suffix gradients, masks, prompt independence, mixed losses, independent EMAs, checkpoint resume, and distributed normalization.
- The GPU check uses the pinned real Qwen model and fitted projectors, including a prompt longer than 257 tokens, cache/live parity, different extraction chunks, compiled small-model joint updates, and generation.
- [Testing](docs/validation.md) describes the CPU and GPU checks.

## Attribution

Built from [ELF](https://github.com/lillian039/ELF). The ELF MIT license is retained. [Third-party notices](docs/third_party.md) identify Qwen, Muon, DiffusionNFT, Math-Verify, and EvalPlus. No model weights, datasets, caches, or experiment outputs are included.
