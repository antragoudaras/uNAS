#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_egg_low_freq_subject_id2_cropped_constrained_128_kb_70_epochs_10_percent
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_low_freq_cropped_subject_id2_no_pruning_new_setup_128kb_70_epochs_10_percent

sbatch "$SRC_DIR"/train_egg_aging_evo_subj2_128kb_cropped_10perc_low_freq.sbatch --name "$JOB_NAME"