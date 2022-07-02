#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=20G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --job-name="uNAS_egg_dataset_cropped_subject_id7_no_pruning_new_setup_256kb_130_epochs_10_percent"
#SBATCH --output=./results_egg_cropped_subject_id7_constrained_128_kb_130_epochs_10_percent/%x-%j-slurm.out
#SBATCH --error=./results_egg_cropped_subject_id7_constrained_128_kb_130_epochs_10_percent/%x-%j-slurm.err

set -e
module purge
module load gcc/11.1.0  
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 
# Load software stack
conda activate ./env

python driver.py configs/cnn_egg_aging_subj7_128kb_cropped_10perc.py "$@"