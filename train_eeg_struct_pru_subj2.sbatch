#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --job-name="uNAS_egg_dataset_subject_id2_agingevo_struct_pruning_no_bounds"
#SBATCH --output=./results_eeg_struct_pru_subject_id2/%x-%j-slurm.out
#SBATCH --error=./results_eeg_struct_pru_subject_id2/%x-%j-slurm.err

set -e
module purge
module load gcc/11.1.0  
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 
# Load software stack
conda activate ./env

python driver.py configs/cnn_egg_struct_pru_subj2.py "$@"