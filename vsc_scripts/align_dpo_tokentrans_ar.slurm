#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# module swap cluster/dodrio/gpu_rome_a100_80
# sbatch vsc_scripts/align_dpo_tokentrans.slurm

#SBATCH --job-name=llama-tokentrans-dpo

#SBATCH --mem=200G                   # amount of memory requested per node
#SBATCH --time 72:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./.output/%x-%j.out # output file name

#PBS -l nodes=1:gpus=4
#PBS -A 2023_079

# Enable for A100
export FI_PROVIDER="efa"

export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INFO
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval "$(conda shell.bash hook)"
conda activate /dodrio/scratch/projects/2023_079/anthony_env/handbook

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes 4 \
    scripts/run_dpo.py recipes/llama-2-7b-nl-tokentrans/dpo/config_full.yaml

