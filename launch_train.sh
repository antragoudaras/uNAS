#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/uNAS

JOB_NAME=first-uNAS-cnn_mnist_struct_pru
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

sbatch --job_name "$JOB_NAME" "$SRC_DIR"/train.sbatch --name "$JOB_NAME"