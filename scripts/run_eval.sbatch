#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Longhorn v100 nodes
#
#   *** Multiple Serial Jobs in v100 Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Longhorn login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#----------------------------------------------------
#SBATCH -J rank1           # Job name
#SBATCH -o slurm_logs/rank1_eval.o%j       # Name of stdout output file
#SBATCH -e slurm_logs/rank1_eval.e%j       # Name of stderr error file
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00                       # Run time (hh:mm:ss)
#SBATCH -p rtx                            # Queue
#SBATCH -A ASC21003                       # Allocation

# Launch serial code to specific GPUs...

ITER=50000

CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval.sh  pl_experiments/default_default/0_1/checkpoints/N-Step-Checkpoint_0_80000.ckpt models/ct_code_summarization/CT-1/config.json 1> slurm_logs/base.out 2> slurm_logs/base.std &
#CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh  pl_experiments/rank1_encoder_agb4_uniform1/default_default/0_0/checkpoints/N-Step-Checkpoint_0_${ITER}.ckpt models/ct_code_summarization/CT-1/rank1_config.json 1> slurm_logs/uniform1.out 2> slurm_logs/uniform1.std &
#CUDA_VISIBLE_DEVICES=2 bash scripts/run_eval.sh  pl_experiments/rank1_encoder_agb4_normal0.05/default_default/0_0/checkpoints/N-Step-Checkpoint_0_${ITER}.ckpt models/ct_code_summarization/CT-1/rank1_config.json 1> slurm_logs/normal0.05.out 2> slurm_logs/normal0.05.std &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh  pl_experiments/full_rank1_wpointer_uniform1/default_default/0_0/checkpoints/N-Step-Checkpoint_0_${ITER}.ckpt models/ct_code_summarization/CT-1/fullrank1pointer_config.json 1> slurm_logs/fullrank1pointer.out 2> slurm_logs/fullrank1pointer.std &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_eval.sh  pl_experiments/rank1_encoder_agb4_normal0.5/default_default/0_0/checkpoints/N-Step-Checkpoint_0_${ITER}.ckpt models/ct_code_summarization/CT-1/rank1_config.json 1> slurm_logs/normal0.5.out 2> slurm_logs/normal0.5.std &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_eval.sh  pl_experiments/full_rank1_uniform1/default_default/0_0/checkpoints/N-Step-Checkpoint_0_${ITER}.ckpt models/ct_code_summarization/CT-1/fullrank1_config.json 1> slurm_logs/fullrank1.out 2> slurm_logs/fullrank1.std &
wait
# ---------------------------------------------------
