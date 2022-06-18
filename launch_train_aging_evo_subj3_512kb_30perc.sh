#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_egg_subject_id3_constrained_512_kb_75_epochs_30_percent
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=uNAS_egg_dataset_subject_id3_no_pruning_new_setup_512kb_75_epochs_30_percent

sbatch "$SRC_DIR"/train_egg_aging_evo_subj3_512kb_30perc.sbatch --name "$JOB_NAME"