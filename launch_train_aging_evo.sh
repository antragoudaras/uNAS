#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_egg_subject_id3
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_subject_id3_agingevo_nopruning

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_egg_aging_evo.sbatch --name "$JOB_NAME"