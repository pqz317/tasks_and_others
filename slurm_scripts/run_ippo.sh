#!/bin/bash
#SBATCH --job-name=ippo

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=2880

module load singularity

mkdir -p /gscratch/walkerlab/patrick/tasks_and_others/results/.wandb_cache
mkdir -p /gscratch/walkerlab/patrick/tasks_and_others/results/.wandb_config

if [ "$#" -gt 0 ]; then
    printf 'Forwarding args to ippo_general.py:'
    printf ' %q' "$@"
    printf '\n'
else
    echo "Forwarding args to ippo_general.py: <none>"
fi

singularity exec \
    --writable-tmpfs \
    --nv \
    --pwd /src/tasks_and_others \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env TF_FORCE_GPU_ALLOW_GROWTH=true \
    --env SLURM_JOB_ID=${SLURM_JOB_ID} \
    --env XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false\ --xla_gpu_enable_cudnn_frontend=false \
    --env WANDB_DIR=/src/tasks_and_others/results \
    --env WANDB_CACHE_DIR=/src/tasks_and_others/results/.wandb_cache \
    --env WANDB_CONFIG_DIR=/src/tasks_and_others/results/.wandb_config \
    --env-file /gscratch/walkerlab/patrick/tasks_and_others/.env \
    --bind /gscratch/walkerlab/patrick/tasks_and_others:/src/tasks_and_others \
    /gscratch/walkerlab/patrick/singularity/tasks_and_others_image.sif \
    /bin/python -u /src/tasks_and_others/baselines/CEC/ippo_general.py \
    ENV_NAME=overcooked \
    NUM_STEPS=256 \
    TOTAL_TIMESTEPS=1e9 \
    FC_DIM_SIZE=256 \
    GRU_HIDDEN_DIM=256 \
    GRAPH_NET=True \
    LSTM=True \
    ENTITY=pqz317-university-of-washington \
    PROJECT=sep-rep-learning \
    WANDB_MODE=online \
    "$@"
