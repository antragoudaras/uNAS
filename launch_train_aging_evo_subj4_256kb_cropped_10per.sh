#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/rerun2_results_egg_cropped_subject_id4_constrained_256_kb_130_epochs_10_percent
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=rerun2_uNAS_egg_dataset_cropped_subject_id4_no_pruning_new_setup_256kb_130_epochs_10_percent

sbatch "$SRC_DIR"/train_egg_aging_evo_subj4_256kb_cropped_10perc.sbatch --name "$JOB_NAME"