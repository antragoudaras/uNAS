#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=20G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --job-name="rerun2_uNAS_egg_dataset_cropped_subject_id4_no_pruning_new_setup_256kb_130_epochs_10_percent"
#SBATCH --output=./rerun2_results_egg_cropped_subject_id4_constrained_256_kb_130_epochs_10_percent/%x-%j-slurm.out
#SBATCH --error=./rerun2_results_egg_cropped_subject_id4_constrained_256_kb_130_epochs_10_percent/%x-%j-slurm.err

set -e
module purge
module load gcc/11.1.0  
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11.2.2 
# Load software stack
conda activate ./env

python driver.py configs/cnn_egg_aging_subj4_256kb_cropped_10perc.py "$@"