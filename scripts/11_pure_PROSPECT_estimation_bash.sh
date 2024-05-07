#!/bin/bash
#SBATCH --job-name=run11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/fji7/transfer_learning_paper/6_log_files/11_pure_PROSPECT_estimation_log.out

# Activate the conda environment
source /software/fji7/miniconda3/bin/activate /software/fji7/miniconda3/envs/Fujiang_envs

trait=$1
type=$2

# Run the script
python /scratch/fji7/transfer_learning_paper/4_src_code/11_pure_PROSPECT_estimation.py $trait $type