#!/bin/bash
# Save this as run_GliODIL.sh

# Check for directory argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1

# Source the configuration file for static options
source config.sh

# Find files in the given directory with either .nii or .nii.gz extension
seg_path=$(realpath $(find "$directory" -type f \( -name '*segm*.nii' -o -name '*segm*.nii.gz' \) -print -quit))
wm_path=$(realpath $(find "$directory" -type f \( -name '*_wm_*.nii' -o -name '*_wm_*.nii.gz' \) -print -quit))
gm_path=$(realpath $(find "$directory" -type f \( -name '*_gm_*.nii' -o -name '*_gm_*.nii.gz' \) -print -quit))
pet_path=$(realpath $(find "$directory" -type f \( -name '*_pet_*.nii' -o -name '*_pet_*.nii.gz' \) -print -quit))

# PDE discounts to iterate over
lambda_pde_multipliers=("1.0")

# CUDA devices you want to use
gpus=("0")


# Make sure to restart from the first GPU when the number of discounts is larger than the number of GPUs
len=${#gpus[@]}
i=0

# Check if PET file exists and set the flag accordingly
pet_usage=""
if [ -z "$pet_path" ]; then
    use_pet_flag="_pet_"
fi

# Iterate over PDE multipliers
for pde_multiplier in "${lambda_pde_multipliers[@]}"; do
    code="x"
    cmd="USEGPU=1 CUDA_VISIBLE_DEVICES=${gpus[$i]} ./GliODIL.py \
        --outdirectory \"$directory/GliODIL_res\" \
        --optimizer $OPTIMIZER \
        --postfix ${use_pet_flag}${POSTFIX}_PDE${pde_multiplier}_ \
        --lambda_pde_multiplier $pde_multiplier \
        --Nt $Nt --Nx $Nx --Ny $Ny --Nz $Nz \
        --days $DAYS \
        --history_every $HISTORY_EVERY \
        --report_every $REPORT_EVERY \
        --epochs $EPOCHS \
        --plot_every $PLOT_EVERY \
        --save_solution $SAVE_SOLUTION \
        --final_print $FINAL_PRINT \
        --code $code \
        --multigrid $MULTIGRID \
        --save_forward $SAVE_FORWARD \
        --save_forward2 $SAVE_FORWARD2 \
        --initial_guess $INITIAL_GUESS \
        --seg_path \"$seg_path\" \
        --wm_path \"$wm_path\" \
        --gm_path \"$gm_path\" \
        --pet_path \"$pet_path\""
    
    # Print the command
    echo $cmd
    
    # Execute the command
    eval $cmd &
    
    i=$(( (i+1) % len ))
    # Wait for all background processes to finish before starting the next round of jobs
    if [ $i -eq 0 ]; then
        wait
    fi
done

wait
