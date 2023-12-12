import subprocess
import numpy as np
import os
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Run script with custom iterations")
parser.add_argument('--N', type=int, default=5, help='Number of iterations')

# Parse arguments
args = parser.parse_args()

# Use the argument
num_iterations = args.N
np.random.seed(42000)
base_output_dir = "synthetic_runs1T/synthetic1T"
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the output directory relative to the script directory
base_output_dir = os.path.join( "synthetic_runs1T", "synthetic1T")

processes = []  # Store the processes so we can check on them later

for i in range(num_iterations):
    # Generate your parameters using a uniform distribution
    Dw = np.random.uniform(low=0.035, high=0.2)
    rho = np.random.uniform(low=0.035, high=0.2)
    RatioDw_Dg = np.random.uniform(low=10, high=30)
    th_necro = np.random.uniform(low=0.70, high=0.85)
    th_up = np.random.uniform(low=0.45, high=0.60)
    th_down = np.random.uniform(low=0.15, high=0.35)

    
    # Generate tumor center positions as percentages
    NxT1_pct = np.random.uniform(low=0.3, high=0.5)
    NyT1_pct = np.random.uniform(low=0.3, high=0.5)
    NzT1_pct = np.random.uniform(low=0.3, high=0.5)

    # Construct the output directory with index
    output_dir = f"{base_output_dir}_run{i}"
    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # Construct the command
    command = [
        'python', 'synthetic_generator.py',
        '--Dw', str(Dw),
        '--rho', str(rho),
        '--RatioDw_Dg', str(RatioDw_Dg),
        '--th_necro', str(th_necro),
        '--th_up', str(th_up),
        '--th_down', str(th_down),
        '--NxT1_pct', str(NxT1_pct),
        '--NyT1_pct', str(NyT1_pct),
        '--NzT1_pct', str(NzT1_pct),
        '--NxT2_pct', str(-1),
        '--NyT2_pct', str(-1),
        '--NzT2_pct', str(-1),
        '--NxT3_pct', str(-1),
        '--NyT3_pct', str(-1),
        '--NzT3_pct', str(-1),
        '--gm_path', os.path.join("precomputed", "sGM_192_192_192.npy"),
        '--wm_path', os.path.join("precomputed", "sWM_192_192_192.npy"),
        '--out_dir', output_dir
        
    ]
    # Convert the command list to a string and print
    command_str = ' '.join(command)
    print('Running:', command_str)
    print('The process will take some minutes.')

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    processes.append((i, process))


# Wait for all processes to complete
for i, process in processes:
    stdout, stderr = process.communicate()  # This waits for the process to finish
    if process.returncode != 0:
        print(f"Error in run {i}:", stderr)
    else:
        print(f"Output from run {i}:", stdout)
# %%
