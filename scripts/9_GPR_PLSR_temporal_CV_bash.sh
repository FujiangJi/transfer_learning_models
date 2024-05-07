#!/bin/bash
#SBATCH --job-name=run10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/fji7/transfer_learning_paper/6_log_files/9_GPR_PLSR_temporal_CV_log.out

# Activate the conda environment
source /software/fji7/miniconda3/bin/activate /software/fji7/miniconda3/envs/Fujiang_envs

# Run the script
python /scratch/fji7/transfer_learning_paper/4_src_code/9_GPR_PLSR_temporal_CV.py