# Joint fine-tuning from released weights

All six [released models](models.md) can initialize joint training of ELF and the prompt encoder. The selected pair becomes the new raw weights, with fresh optimizers, scheduler counters, data cursor, and independent EMA .99/.999/.9999 states. The frozen token embedding and teacher projectors remain fixed.

## Download and prepare data

Install CEDR following the README. Choose the matching training recipe and download one model outside the code checkout:

```bash
recipe=configs/math_l.yaml
artifacts=/path/to/cedr-artifacts
teacher=/path/to/Qwen3-4B-Instruct-2507
model="$artifacts/math-pre"

cedr download --model math-l-pre-nft --output "$model"
cedr prepare --config "$recipe" --input "$artifacts/train.jsonl" \
  --tokenizer "$model/shared/tokenizer" --output "$artifacts/data"
```

Use `configs/gsm8k_b.yaml` for either GSM8K-B model and `configs/oci_l.yaml` for either OCI-L model. The [pipeline guide](pipeline.md) describes the input JSONL schema, answer formatting, and length eligibility. Training rows need both prompts and solutions. Use your own training population; benchmark preparation creates inference-only data.

Download `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554` for target extraction. The released embedding table supplies the prompt encoder, but extracting contextual teacher targets requires the full teacher Transformer. No teacher sampling or projector fitting is needed.

Each download includes a model-specific `representation.pt`: layer means, effective whitened encoders, reconstruction matrices, teacher identity, and compatible model IDs. Whitening is already folded into the encoders. Keep it paired with the model; identically sized representations from different families are not interchangeable.

## Train with live features

```bash
torchrun --standalone --nproc-per-node=2 -m cedr train joint \
  --config "$recipe" --init-weights "$model" \
  --data "$artifacts/data" --features live --teacher "$teacher" \
  --qwen-chunk 8 --micro-batch 32 --output "$artifacts/joint-finetune"
```

The package supplies the architecture, including the actual time embedders, prompt encoder, and frozen embedding. The recipe supplies optimization, additional joint-training epochs (`stages.joint_epochs`), and loss settings. Copy/edit the recipe to choose your fine-tuning duration or learning rate. Defaults retain effective batch 512, synchronous clocks, mixed flow/decoder CE, and gradient clipping at 1. OCI enables the configurable 2:1 gradient-balanced prompt-MSE term; math disables it.

The package already contains one selected model/prompt pair. Do not pass EMA selectors or `--prompt-init` when using `--init-weights`. Initialization begins a new training run at update zero. `--max-updates N` can bound a first check without changing the configured horizon. Epoch endpoints must meet the existing complete-accumulation requirement.

Prepared data is checked against the task, pinned teacher/tokenizer revision, canvas, and, for newly prepared data, tokenizer fingerprint. An explicit `--representation` or `--embedding` override must match the package hash.

## Train with a feature cache

```bash
cedr cache --config "$recipe" --data "$artifacts/data" \
  --teacher "$teacher" --representation "$model/representation.pt" \
  --batch-rows 8 --output "$artifacts/cache"

torchrun --standalone --nproc-per-node=2 -m cedr train joint \
  --config "$recipe" --init-weights "$model" \
  --data "$artifacts/data" --features cache --cache "$artifacts/cache" \
  --micro-batch 32 --output "$artifacts/joint-finetune"
```

Cache construction uses frozen causal Qwen, FP32 projection, and BF16 targets. Cached training requires no live Qwen Transformer. Cache metadata binds both the prepared dataset and representation hash. OCI's auxiliary prompt targets come from the same features.

## Resume your fine-tuning run

```bash
torchrun --standalone --nproc-per-node=2 -m cedr train joint \
  --config "$recipe" --resume "$artifacts/joint-finetune/latest.pt" \
  --data "$artifacts/data" --features cache --cache "$artifacts/cache" \
  --micro-batch 32 --output "$artifacts/joint-finetune"
```

Use the same feature backend, training recipe, data, and GPU topology as the saved run; the example resumes the cached command above. For a live run, retain `--features live --teacher "$teacher"`. Omit `--init-weights` on resume. The saved effective architecture, model/prompt states, optimizers, EMAs, RNG, sampler position, and any MSE-balancing state are restored. The checkpoint records embedding, representation, and tokenizer locations; if relocated, supply their new paths, preserving their identities.

The existing `--init` plus `--prompt-init` stage transition remains available for training the complete pipeline with mature ELF optimizer state. `--init-weights` is the separate route for starting fine-tuning from released weights.
