#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_egg_cropped_subject_id1_constrained_256_kb_130_epochs_20_percent
LOAD_FROM="$PROJECT_DIR"/artifacts/cnn_egg_cropped/uNAS_egg_dataset_cropped_subject_id1_no_pruning_new_setup_256kb_130_epochs_20_percent_agingevosearch_state.pickle
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_cropped_subject_id1_no_pruning_new_setup_256kb_130_epochs_20_percent

sbatch "$SRC_DIR"/train_egg_aging_evo_subj1_256kb_cropped.sbatch --name "$JOB_NAME" --load-from "$LOAD_FROM"