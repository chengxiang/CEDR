# Slurm example

The library has no Slurm dependencies. `train.sh` shows single-node torchrun, a separate scheduler resource request, and optimizer-boundary preemption saves. Set your cluster's partition/account/QoS at submission and load the appropriate CUDA module and Python environment.

```bash
mkdir -p logs
export CEDR_RECIPE=configs/math_l.yaml
export CEDR_DATA=/storage/data
export CEDR_OUTPUT=/storage/flow
export CEDR_STAGE=flow
export CEDR_GPUS=2
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT \
  examples/slurm/train.sh --features cache --cache /storage/cache
```

Set `--gres`, `CEDR_GPUS`, CPU/RAM limits, and time together. The wrapper automatically resumes `latest.pt` in the explicitly selected output directory. For stage transitions pass the initial ELF/prompt artifacts as documented in the pipeline guide. The numerical runtime records the resulting world size, local microbatch, and accumulation. No cluster launch is performed by package installation or tests.
