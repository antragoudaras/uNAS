#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=continue_36h-uNAS-cnn_mnist_struct_pru

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train.sbatch --load-from /ibex/scratch/tragoua/tensorflow-uNAS/uNAS/artifacts/cnn_mnist/continue-uNAS-cnn_mnist_struct_pru_agingevosearch_state..pickle --name "$JOB_NAME"