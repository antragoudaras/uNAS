#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_egg_subject_id8_constrained_256_kb
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_subject_id8_no_pruning_new_setup_256kb

sbatch "$SRC_DIR"/train_egg_aging_evo_subj8_256kb.sbatch --name "$JOB_NAME"