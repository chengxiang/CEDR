# Sampling conventions

`cedr generate` supports three samplers. Released presets use `dmd2`; `--sampler` overrides the preset. A configuration without a sampler field uses ODE.

| Sampler | Option | Transition | Terminal state |
|---|---|---|---|
| ODE | `--sampler ode` | Euler with the local clock derivative | Final Euler state |
| EDM-style SDE | `--sampler edm --edm-churn 0.7` | Backward noising followed by finite local-time interpolation | Final clean prediction |
| DMD2-style re-noising | `--sampler dmd2 --renoise-eta 1` | Mix the clean prediction with fresh Gaussian noise at the next local time | Final clean prediction |

These names describe sampling rules; they do not imply distillation training. Guidance, recurrent self-conditioning, and the selected model/prompt pair are shared across samplers. Prompt positions are restored after each intervention and update.

All presets use a logit-normal quantile grid with mean -1.5 and standard deviation 0.8, endpoints 0 and 1, and local clocks `t_l = tau ** gamma_l` with powers `(2.5, 2, 1.5)`. Initial noise has standard deviation 2. The ODE velocity denominator is clamped at 0.05 and includes the power-clock derivative.

For DMD2-style sampling, the nonterminal update is `z_next,l = t_next,l * x0_l + (1 - t_next,l) * eta * 2 * epsilon_l`. There are S-1 fresh draws for S denoiser calls. The final call returns its clean prediction without another noise draw.

EDM-style sampling contracts the **quantile index**, using `G(c * r / S)` at interior grid index r, rather than multiplying the current time by c. In each feature group, let t be the current local time, b the backward local time, and a=b/t. Backward noising is `a*z + sqrt(4*(1-a)*(1+a-2*b))*epsilon`. The forward interpolation weight is `(t_next-b)/(1-b)`. The first step uses finite local-time interpolation without applying noise; its random draw is consumed. Interior steps apply S-2 noise interventions. The final step returns the clean prediction without a draw. The parameter c is in [0,1]; c=1 disables backward noise.

## Randomness and precision

Initial states use a CPU generator seeded with `--seed`. Fresh stochastic noise uses a separate generator on the inference device, seeded per batch as `(seed + 1000003 + first_row * 1000003) % (2**63 - 1)`. The first row is the zero-based index in the prepared benchmark. Keep batch size and row order fixed for exact seeded comparisons.

States and learned weights are FP32, model forwards use BF16, and grouped time embeddings are averaged in FP64 before conversion to FP32. One terminal vocabulary decoder call is additional to the denoising-step count. Generation records sampler parameters, random streams, call counts, timing, and endpoint convention. Resume restores the initial-noise stream and derives each batch's fresh stream from its row offset; changing settings requires a new output directory.

```bash
cedr generate --checkpoint artifacts/gsm8k-l-pre --data artifacts/gsm8k-test \
  --sampler edm --edm-churn 0.7 --steps 64 --seed 42 --output artifacts/edm-seed42
```
