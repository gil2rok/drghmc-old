#!/bin/bash
#SBATCH --job-name=drghmc
#SBATCH --array=1-4                         ## launch 4 distinct jobs, each with one task
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --cpus-per-task=1

module -q purge
source ~/mambaforge/bin/activate drghmc
python src/main.py --model_num 2 --chain_num ${SLURM_ARRAY_TASK_ID}
