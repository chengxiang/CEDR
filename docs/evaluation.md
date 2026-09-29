# Generation and benchmark scoring

Install `.[data,math]`, adding `.[code]` for coding benchmarks. Linux bubblewrap (`bwrap`) and enabled user namespaces are required for isolated code scoring.

## GSM8K and MATH500

```bash
cedr download --model math-l-post-nft --output artifacts/math-post
cedr prepare-benchmark --benchmark math500 --checkpoint artifacts/math-post \
  --output artifacts/math500
cedr generate --checkpoint artifacts/math-post --data artifacts/math500 \
  --seed 42 --output artifacts/math-seed42
cedr score --config artifacts/math-post/config.json --data artifacts/math500 \
  --predictions artifacts/math-seed42/predictions.jsonl \
  --workers 8 --output artifacts/math-seed42/scoring
```

For GSM8K, use either GSM8K model ID and `--benchmark gsm8k`. Benchmark preparation pins the dataset revision, validates the complete ordered cohort, and uses only questions to create native Qwen prompts. Gold answers are retained solely for scoring. No training-length cap or truncation is applied to inference prompts.

## HumanEval and MBPP-378

Prepare the combined coding cohort to reproduce the reported batch order and noise stream: 164 HumanEval tasks followed by the specified 378 MBPP tasks. The cohort, including task IDs and source versions, is pinned in `benchmark_specs.json`. Both base and plus tests are scored.

```bash
cedr download --model oci-l-post-nft --output artifacts/oci-post
cedr prepare-benchmark --benchmark coding --checkpoint artifacts/oci-post \
  --output artifacts/coding
cedr generate --checkpoint artifacts/oci-post --data artifacts/coding \
  --seed 42 --output artifacts/coding-seed42
cedr score --config artifacts/oci-post/config.json --data artifacts/coding \
  --predictions artifacts/coding-seed42/predictions.jsonl \
  --benchmark humaneval --evalplus-cache artifacts/coding/evalplus-cache \
  --workers 16 --output artifacts/coding-seed42/humaneval
cedr score --config artifacts/oci-post/config.json --data artifacts/coding \
  --predictions artifacts/coding-seed42/predictions.jsonl \
  --benchmark mbpp --evalplus-cache artifacts/coding/evalplus-cache \
  --workers 16 --output artifacts/coding-seed42/mbpp
```

`prepare-benchmark` also accepts `humaneval` or `mbpp` individually. These use the same tasks; their batch boundaries and seeded noise assignment differ from combined-cohort generation. The scorer selects the requested benchmark from combined predictions and requires complete input coverage.

Official EvalPlus runs with networking disabled and a minimal filesystem. Standard scoring is the default: its pinned sanitizer receives the task entry point without adding aliases. The minimum test timeout is 4 seconds and the reference-time multiplier is 4. Startup errors are reported as failures, not incorrect solutions.

### Optional function-name repair

For the separately reported alias-scored results, add `--function-name-repair` to either coding scoring command. Reuse the same predictions and write to a separate output directory:

```bash
cedr score --config artifacts/oci-post/config.json --data artifacts/coding \
  --predictions artifacts/coding-seed42/predictions.jsonl \
  --benchmark mbpp --evalplus-cache artifacts/coding/evalplus-cache \
  --function-name-repair --workers 16 \
  --output artifacts/coding-seed42/mbpp-alias
```

The [extraction rule](../src/cedr/code_extraction.py) first sanitizes without an entry point. If the expected name is absent and exactly one top-level function is present, it appends `expected_name = generated_name` and sanitizes again using the expected entry point. Existing bindings, parse failures, and zero or multiple top-level functions retain standard extraction. It preserves function bodies and recursive calls and does not use test outcomes or reference solutions to select a function.

Generation, base/plus tests, and execution limits stay the same. `scoring.json` records the extraction mode and alias/change counts; `extraction_audit.jsonl` records each task's decision. Scoring rejects reuse of an output directory across modes. Report alias-scored accuracies separately from standard accuracies. The flag applies only to `score --benchmark`; training rewards use standard extraction.

## Sampling and resumption

Presets are listed in [models.md](models.md). Override `--seed`, `--steps`, `--sccfg`, `--powers`, or `--batch-size` explicitly. Use a distinct output directory for each seed/setting. Initial Gaussian noise uses a seeded CPU generator; the deterministic logit-normal quantile grid uses mean −1.5 and standard deviation .8. Local clock derivatives are included in Euler updates. SCCFG is a learned guidance input, so SCCFG=1 still uses recurrent self-conditioning. CFG=1 requires no unconditional model calls.

Selected weights and ODE states are FP32; denoiser Transformer forwards use BF16. Time embeddings are averaged in FP64. Vocabulary projection and greedy decoding use FP32, with PyTorch matmul precision set to `high`. Math/GSM prompt encoders use FP32 with dynamic padding; coding prompt encoding uses BF16 and padding to a multiple of 32. These settings are recorded in each package configuration.

Generation atomically saves completed predictions, CPU generator state, and timing after each batch. Repeating the same command resumes. Changing model identity, data, selectors, seed, batch size, clocks, guidance, or numerical settings rejects reuse. Reports retain hashes, complete coverage, per-batch timing, and network-call counts. Scoring similarly checks its input contract.

Existing CEDR checkpoint files remain supported using `--config`, `--embedding`, and `--tokenizer`; `--elf-selector` and `--prompt-selector` select independent states. A packaged directory already contains its selected pair. A flow checkpoint can use teacher prompts with `--prompt-source teacher --features live --teacher ... --representation ...`.

## Accuracy definitions

GSM8K uses the canonical numerical extractor. MATH500 uses Math-Verify 0.9.0 with boxed-first extraction, strict equivalence, 6-digit rounding, and a 5-second timeout. Coding uses the pinned official EvalPlus base/plus tests. OCI training rewards use a separate reference-validated native-test population.

A single seeded evaluation measures pass@1. Use matched problem IDs and multiple independent seeds for pass@k or paired bootstrap intervals. Do not pool different protocols. Release-validation scores are reported separately from the paper's eight-seed estimates.
