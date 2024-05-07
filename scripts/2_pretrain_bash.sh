#!/bin/bash
#SBATCH --job-name=run2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/fji7/transfer_learning_paper/6_log_files/2_pretrain_DNN_log.out

# Activate the conda environment
source /software/fji7/miniconda3/bin/activate /software/fji7/miniconda3/envs/Fujiang_envs

# Run the script
python /scratch/fji7/transfer_learning_paper/4_src_code/2_pretrain_DNN.py

