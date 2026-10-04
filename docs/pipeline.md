# Pipeline and data contracts

All paths below are supplied by the user. Run from the repository checkout after installation. The same commands work with `python -m cedr`; `torchrun -m cedr` supplies distributed training.

## 1. Data

Input is UTF-8 JSONL, one training row per solution:

```json
{"id":"unique-row-id","prompt":"What is 1+1?","answer":"One plus one is two. \\boxed{2}","weight":1.0,"metadata":{"gold_answer":"2","subject":"algebra","level":1}}
```

For OCI, keep the original prompt and code whitespace and include `tests`, a nonempty list of native assertion strings. `metadata.gold_answer` supplies the canonical MATH answer for rewards/scoring when `answer` contains a full training solution. Keep benchmark task IDs (`HumanEval/0`, `Mbpp/…`) as `id` for EvalPlus. Never use benchmark tests as training rewards.

`cedr fetch` is a streaming field adapter for Hugging Face datasets. The caller must pin a revision and explicitly select fields, split, and subset. It records the revision and output hash. It does not guess dataset-specific corpus mixtures.

```bash
cedr fetch --dataset ORGANIZATION/DATASET --revision COMMIT \
  --split train --prompt-field problem --answer-field generated_solution \
  --metadata-fields subject level --output "$artifacts/train.jsonl"
```

Prepare training and inference datasets separately:

```bash
cedr prepare --config "$recipe" --input "$artifacts/train.jsonl" \
  --tokenizer "$teacher" --output "$artifacts/data"
cedr prepare --config "$recipe" --input "$artifacts/test.jsonl" \
  --tokenizer "$teacher" --inference --output "$artifacts/test"
```

Preparation uses the native pinned Qwen chat template. Math prompts append `Please reason step by step, and put your final answer within \boxed{}.` GSM calculator annotations are removed and `#### answer` is converted to `\boxed{answer}`. OCI prompts/solutions receive no math formatting. Native assistant content ends before `<|im_end|>`; supervised answer masks include exactly one terminal token and exclude the template's following newline. No row is truncated.

Default eligibility:

| Task | Native training prompt cap | Total training tokens | Inference canvas |
|---|---:|---:|---:|
| GSM8K | 257 | 1024 | 1024, up to 1023 prompt tokens |
| MATH | 257 | 1024 | 1024, up to 1023 prompt tokens |
| OCI | no separate cap | 1024 | 1024, up to 1023 prompt tokens |

The manifests report accepted/rejected counts, input hash, tokenized-file hashes, and distinct prompt count. Standalone prompt training deduplicates by tokenized question prefix. Flow/joint training retains every eligible solution row, including repeated prompts.

MATH training uses eligible OpenMathInstruct-2 train-5M math/augmented-math rows, with train-2M rows for representation fitting. OCI uses eligible OpenCodeInstruct rows. GSM8K accepts reasoning corpora in the normalized schema above. Supply the intended training population and retain its filtering and source manifests.

For OpenMathInstruct-2, explicitly restrict its `problem_source` field with `fetch --filter-field problem_source --filter-values math augmented_math --gold-field expected_answer`. Keep the full source-field manifest when choosing your training population.

## 2. Covariance and learned projectors

`cedr select --input SOURCE.jsonl --output PANEL.jsonl --count 200000 --strata subject level --seed 42` builds an equally allocated stratified panel, redistributing exhausted strata, preserving source order, and attaching inverse inclusion weights. Its receipt records every stratum's population, quota, and selection probability. Choose strata from the metadata fields available in the source dataset.

Prepare a separate `projector-data` directory using the same tokenizer. A sampled population can attach positive inverse-inclusion `weight` values; covariance and teacher CE honor them. Keep sampling seed, strata, inclusion probabilities, and original row IDs with that population. For MATH, use a stratified 200,000-row representation-fitting panel and covariance statistics from its full eligible source population.

```bash
cedr covariance --config "$recipe" --data "$artifacts/projector-data" \
  --teacher "$teacher" --batch-rows 4 --output "$artifacts/covariance.pt"
cedr projectors --config "$recipe" --data "$artifacts/projector-data" \
  --teacher "$teacher" --covariance "$artifacts/covariance.pt" \
  --batch-rows 4 --output "$artifacts/projectors"
```

Streaming FP64 centered moments cover assistant-content tokens only. Save the eigensolver's ordering and signs; clamp covariance eigenvalues at `1e-5`. Each learned encoder is `C^(-1/2) Q`, with `Q` obtained by differentiable canonical QR. Its reconstruction matrix is independently trainable. Three independent interventions at layers 16/24/32 optimize full-vocabulary teacher soft CE through their frozen suffixes. Hidden position `j` predicts content token `j+1`; the first answer token, EOS, prompt, and padding are excluded as CE targets. The teacher's parameters are frozen while suffix differentiation into the reconstruction remains enabled.

Projectors use 10 epochs, Muon at 0.001, clipping at 1, and terminal EMA .999 export. Set the projector batch size explicitly with `--batch-rows` and record it with the run configuration.

Optional prepared activations avoid repeated clean captures:

```bash
cedr activations --config "$recipe" --data "$artifacts/projector-data" \
  --teacher "$teacher" --batch-rows 4 --output "$artifacts/activations"
cedr projectors --config "$recipe" --data "$artifacts/projector-data" \
  --teacher "$teacher" --covariance "$artifacts/covariance.pt" \
  --activation-cache "$artifacts/activations" --batch-rows 4 \
  --output "$artifacts/projectors-prepared"
```

Prepared activations and live activations use the same independent-intervention objective. The suffix still runs live; the implementation reruns the frozen prefix to supply untouched context. This trades speed for a single native causal-forward path.

## 3. Optional feature cache

```bash
cedr cache --config "$recipe" --data "$artifacts/data" --teacher "$teacher" \
  --representation "$artifacts/projectors/representation.pt" \
  --batch-rows 8 --shard-rows 256 --output "$artifacts/cache"
```

Shards contain packed BF16 targets with row offsets, hashes, and dataset/representation identities. Invalid canvas positions are zero. Publication is atomic per shard; rerunning reuses verified completed shards.

For the commands below define either backend:

```bash
# Cached backend (default):
features=(--features cache --cache "$artifacts/cache" \
          --representation "$artifacts/projectors/representation.pt")
# Or, without creating a feature cache:
features=(--features live --teacher "$teacher" \
          --representation "$artifacts/projectors/representation.pt" --qwen-chunk 8)
```

`--embedding` accepts the pinned local Qwen snapshot directory or its embedding safetensors shard. Cached prompt/joint/NFT training loads only the embedding table, not the Qwen transformer. Live mode loads frozen Qwen for target extraction. Both backends preserve row order and training RNG streams; changing `--qwen-chunk` does not alter the optimization batch.

## 4. Supervised stages

To start from a released model pair, use `train joint --init-weights MODEL_DIR` with fresh optimizers as described in the [fine-tuning guide](finetuning.md). The stage-transition commands below preserve optimizer state when training the full pipeline.

```bash
torchrun --standalone --nproc-per-node=2 -m cedr train flow \
  --config "$recipe" --data "$artifacts/data" "${features[@]}" \
  --micro-batch 32 --output "$artifacts/flow"

torchrun --standalone --nproc-per-node=2 -m cedr train prompt \
  --config "$recipe" --data "$artifacts/data" "${features[@]}" \
  --embedding "$teacher" --output "$artifacts/prompt"

torchrun --standalone --nproc-per-node=2 -m cedr train joint \
  --config "$recipe" --data "$artifacts/data" "${features[@]}" \
  --embedding "$teacher" --init "$artifacts/flow/final.pt" \
  --prompt-init "$artifacts/prompt/final.pt" --prompt-selector .999 \
  --micro-batch 32 --output "$artifacts/joint"
```

Effective supervised batch is 512. Flow/joint accumulation is computed from GPU count and local microbatch. Prompt imitation defaults to 512/world prompts per GPU and accumulation 1; its final partial batch is normalized by the number of real prompts across ranks. Flow/joint drops only incomplete global microbatches; accumulation may cross epoch boundaries. Changing topology can change the dropped tail. Record the GPU count and microbatch size when comparing runs.

The supervised flow/decoder loss is the per-microbatch answer-token mean, averaged across accumulation and ranks. Decoder mode is drawn independently per row with probability .2; self-conditioning targets are detached. Clean prompt restoration and the dedicated decoder-mode tokens remain active. Do not replace this with independently averaged flow and CE losses.

Raw weights and EMA .99/.999/.9999 are independent. Joint initialization restores mature ELF optimizer state and all its EMAs, then clones the selected prompt EMA into prompt raw/EMAs with fresh prompt optimizers. OCI adds a prompt-only MSE pass whose accumulated, GPU-averaged gradient is balanced against downstream prompt gradients using .99-smoothed norms at target ratio 2. The ELF gradient is unchanged; combination precedes prompt clipping.

## 5. NFT

Prepare a reward population with gold answers. OCI must pass native reference tests before entering the population:

```bash
cedr validate-rewards --config "$recipe" --data "$artifacts/reward-candidates" \
  --output "$artifacts/reward-validation"
cedr prepare --config "$recipe" --input "$artifacts/reward-validation/eligible.jsonl" \
  --tokenizer "$teacher" --output "$artifacts/reward-data"
```

Build a matching cache for `reward-data` or use live features. The large supervised cache cannot be paired with a different dataset manifest. Then:

```bash
torchrun --standalone --nproc-per-node=4 -m cedr train nft \
  --config "$recipe" --data "$artifacts/reward-data" --features live \
  --teacher "$teacher" --representation "$artifacts/projectors/representation.pt" \
  --embedding "$teacher" --init "$artifacts/joint/final.pt" \
  --reward-validation "$artifacts/reward-validation/reference_validation.json" \
  --micro-batch 8 --output "$artifacts/nft"
```

Math tasks may omit the OCI receipt. Use a fixed source population; the sampler chooses unique prompt IDs with their first retained gold solution. The default is 24 groups, 15 generated endpoints plus one gold, and 8 time/noise draws per retained endpoint. Gold/correct/incorrect rewards are 1/.75/0. All-generated-correct groups are discarded before reward normalization. Center rewards within group, divide by the retained raw-tier population standard deviation, clip advantages to ±5, and map to [0,1]. The loss uses one global update-wide token denominator, with beta 1 and reference weight .1.

Generation uses the old policy. Gold targets come from the feature provider; generated endpoints stay diffusion outputs. NFT starts fresh optimizers from independently selected supervised ELF/prompt EMAs, freezes decoder-only parameters, and creates independent feature-group time embedders initialized identically from the supervised scalar embedder. NFT always disables auxiliary prompt MSE. Old-policy decay is `min(0.001 × update, 0.5)` before update 400 and .4 thereafter.

## Resume, exports, and bounded checks

Use the same command and output directory plus `--resume /path/to/latest.pt`. You need not repeat `--init`/`--prompt-init` on ordinary resume. Rank RNG, sampler cursor, all model/EMA/optimizer state, constant-LR scheduler metadata, and MSE norm EMAs are restored. Resume requires the same GPU topology. Optimizer state is replicated across ranks.

`--max-updates N` runs N additional updates without modifying the recipe's stage horizon. `--no-compile` is useful for short debugging checks. SIGUSR1/SIGTERM requests an atomic optimizer-boundary save. Training runs save initialization, permanent half-epoch checkpoints, rolling saves every 500 updates, and final state. Scheduler requeue wiring is external.

```bash
cedr export --input "$artifacts/joint/final.pt" --output "$artifacts/joint-weights.pt"
```

This retains raw weights and all independently keyed EMAs while removing optimizers. To export one selected pair, pass both `--elf-selector` and `--prompt-selector`, then use `raw/raw` when loading the resulting selected-weight export. The examples use weights produced by your own training runs. Released model pairs and matched representations are available on [Hugging Face](https://huggingface.co/xc91/CEDR).
