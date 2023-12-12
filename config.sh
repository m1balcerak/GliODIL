#!/bin/bash
# config.sh - Configuration for GliODIL program execution

# CUDA device configuration
export CUDA_VISIBLE_DEVICES="0"

# Optimization and program settings
export OPTIMIZER="adamn"
export POSTFIX=""
export Nt="192"
export Nx="48"
export Ny="48"
export Nz="48"
export DAYS="100"
export HISTORY_EVERY="1000"
export REPORT_EVERY="1000"
export EPOCHS="9000"
export PLOT_EVERY="3000"
export SAVE_SOLUTION="y"
export FINAL_PRINT="y"
export MULTIGRID="1"
export SAVE_FORWARD="odil_res"
export SAVE_FORWARD2="full_trim_Gauss"
export INITIAL_GUESS="forward_character_dice_breaking"
