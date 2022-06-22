#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_egg_subject_id1_cropped_constrained_512_kb_130_epochs_10_percent_pru
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_cropped_subject_id1_no_pruning_new_setup_512kb_130_epochs_10_percent_pru

sbatch "$SRC_DIR"/train_egg_aging_evo_subj1_512kb_cropped_10perc_pru.sbatch --name "$JOB_NAME" --load-from "$SRC_DIR"/artifacts/cnn_egg_cropped/uNAS_egg_dataset_cropped_subject_id1_no_pruning_new_setup_512kb_130_epochs_10_percent_pru_agingevosearch_state.pickle