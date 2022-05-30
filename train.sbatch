#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=0 
#SBATCH --partition=batch
#SBATCH --constraint=cpu_intel_platinum_8260 
#SBATCH --mail-type=ALL
#SBATCH --output=../results/%x/%j-slurm.out
#SBATCH --error=../results/%x/%j-slurm.err

# Load software stack
conda activate ../env

python driver.py configs/cnn_mnist_struct_pru.py "$@"