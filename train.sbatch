#!/bin/bash --login
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem-per-gpu=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=../results/%x/%j-slurm.out
#SBATCH --error=../results/%x/%j-slurm.err

# Load software stack
conda activate ../env

python driver.py configs/cnn_mnist_struct_pru.py "$@"