#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_eeg_struct_pru_subject_id2
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_subject_id2_agingevo_struct_pruning_no_bounds

sbatch "$SRC_DIR"/train_eeg_struct_pru_subj2.sbatch --name "$JOB_NAME"