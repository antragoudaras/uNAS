#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=20G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --job-name="uNAS_BCI_IV_2a_low_freq_cropped_subject_id1_no_pruning_new_setup_128kb_70_epochs_10_percent_save_models"
#SBATCH --output=./results_BCI_IV_2a_low_freq_subject_id1_cropped_constrained_128_kb_70_epochs_10_percent_save_models/%x-%j-slurm.out
#SBATCH --error=./results_BCI_IV_2a_low_freq_subject_id1_cropped_constrained_128_kb_70_epochs_10_percent_save_models/%x-%j-slurm.err

set -e
module purge
module load gcc/11.1.0  
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 
# Load software stack
conda activate ./env

python driver.py configs/cnn_egg_aging_subj1_128kb_cropped_10perc_low_freq.py "$@"