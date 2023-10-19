#!/bin/bash
#SBATCH --job-name=funnel20             # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (1 task per job)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task (1 core)
#SBATCH --output=logs/slurm_%A_%a.out   # Standard output file (%A is the job ID, %a is the task ID)
#SBATCH --error=logs/slurm_%A_%a.err    # Standard error file (%A is the job ID, %a is the task ID)
#SBATCH --array=1-40                    # Specify the range for the job array (1 to N). SLURM treats the job array as a single job with multiple tasks.

module -q purge
source ~/mambaforge/bin/activate drghmc
mpirun python src/main.py 
python src/main.py --model_num 4 --chain_num ${SLURM_ARRAY_TASK_ID} # TODO: delete everything from ceph/drghmc/res/PDB_04!