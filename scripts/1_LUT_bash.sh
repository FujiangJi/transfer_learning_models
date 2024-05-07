#!/bin/bash
#SBATCH --job-name=run1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/fji7/transfer_learning_paper/6_log_files/1_LUT_construction_log.out

# Activate the conda environment
source /software/fji7/miniconda3/bin/activate /software/fji7/miniconda3/envs/Fujiang_envs
# Run the script
python /scratch/fji7/transfer_learning_paper/4_src_code/1_LUT_construction.py