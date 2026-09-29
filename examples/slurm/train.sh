#!/usr/bin/env bash
#SBATCH --job-name=cedr-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/cedr-%j.out
set -euo pipefail
# Create logs/ before sbatch. Supply partition/account/QoS on the sbatch command.
# Activate your environment and CUDA module before submission, or do so here.
: "${CEDR_RECIPE:?Set CEDR_RECIPE}"
: "${CEDR_DATA:?Set CEDR_DATA}"
: "${CEDR_OUTPUT:?Set CEDR_OUTPUT}"
: "${CEDR_STAGE:?Set CEDR_STAGE to flow, prompt, joint, or nft}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
processes=${CEDR_GPUS:-2}
python_bin=${CEDR_PYTHON:-python}
resume=()
if [[ -f "$CEDR_OUTPUT/latest.pt" ]]; then
    resume=(--resume "$CEDR_OUTPUT/latest.pt")
fi
interrupted=0
launcher_pid=
request_save() {
    interrupted=1
    # torchrun's Python workers install the optimizer-boundary signal handler.
    # Signal workers rather than terminating the elastic launcher itself.
    if [[ -n "$launcher_pid" ]]; then
        pkill -USR1 -P "$launcher_pid" || true
    fi
}
trap request_save USR1 TERM
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node="$processes" \
    -m cedr train "$CEDR_STAGE" --config "$CEDR_RECIPE" --data "$CEDR_DATA" \
    --output "$CEDR_OUTPUT" "${resume[@]}" "$@" &
launcher_pid=$!
set +e
wait "$launcher_pid"
status=$?
if [[ "$interrupted" == 1 ]]; then
    wait "$launcher_pid"
    status=$?
    if [[ "$status" == 0 && -f "$CEDR_OUTPUT/latest.pt" ]]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
fi
exit "$status"
