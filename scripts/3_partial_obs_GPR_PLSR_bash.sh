#!/bin/bash
#SBATCH --job-name=run3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/fji7/transfer_learning_paper/6_log_files/3_partial_obs_GPR_PLSR_log.out

# Activate the conda environment
source /software/fji7/miniconda3/bin/activate /software/fji7/miniconda3/envs/Fujiang_envs

trait=$1
type=$2

# Run the script
python /scratch/fji7/transfer_learning_paper/4_src_code/3_partial_obs_GPR_PLSR.py $trait $type