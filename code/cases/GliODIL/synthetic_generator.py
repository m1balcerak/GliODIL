# %%
#example usage: python synthetic_generator3T.py --Dw 0.12 --rho 0.05 --RatioDw_Dg 20 --gm_path "/path/to/gm_file.npy" --wm_path "/path/to/wm_file.npy" --out_dir "/path/to/output_directory"

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import colorConverter
import copy
from scipy import ndimage
from scipy.ndimage import zoom
plt.rcParams["font.family"] = "cursive"
affine = np.eye(4)

#%%
def find_best_z(data):
    # This function should determine the best Z value.
    # Replace this with your actual implementation if different.
    return np.argmax(np.sum(data, axis=(0, 1)))
# Define command-line arguments
parser = argparse.ArgumentParser(description='Generate FDM solutions for a tumor growing in a brain.')
parser.add_argument('--Dw', type=float, required=True, help='Water diffusion coefficient')
parser.add_argument('--rho', type=float, required=True, help='Tumor cell density')
parser.add_argument('--RatioDw_Dg', type=float, required=True, help='Ratio of Dw to Dg')
parser.add_argument('--gm_path', type=str, required=True, help='Path to the grey matter file')
parser.add_argument('--wm_path', type=str, required=True, help='Path to the white matter file')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save the result')
parser.add_argument('--NxT1_pct', type=float, required=True, help='Percentage position of NxT1')
parser.add_argument('--NyT1_pct', type=float, required=True, help='Percentage position of NyT1')
parser.add_argument('--NzT1_pct', type=float, required=True, help='Percentage position of NzT1')

parser.add_argument('--NxT2_pct', type=float, required=True, help='Percentage position of NxT2')
parser.add_argument('--NyT2_pct', type=float, required=True, help='Percentage position of NyT2')
parser.add_argument('--NzT2_pct', type=float, required=True, help='Percentage position of NzT2')


parser.add_argument('--NxT3_pct', type=float, required=True, help='Percentage position of NxT3')
parser.add_argument('--NyT3_pct', type=float, required=True, help='Percentage position of NyT3')
parser.add_argument('--NzT3_pct', type=float, required=True, help='Percentage position of NzT3')

parser.add_argument('--th_necro', type=float, required=True, help='Threshold for necrotic core')
parser.add_argument('--th_up', type=float, required=True, help='Threshold for enhancing core')
parser.add_argument('--th_down', type=float, required=True, help='Threshold for edema')


args = parser.parse_args()

# Load tissues
sGM = np.load(args.gm_path)
sWM = np.load(args.wm_path)


#%%
def get_pet_signal(u_exact,sigma,scaler,th_necro,th_up,th_down):
    seg = segment_BRATS_volume_cell_distribusion(u_exact,th_necro,th_up,th_down)
    pet = np.where(np.logical_or(seg == 1, seg ==3),u_exact,0)
    if sigma != 0:
        non_zero_mask = np.where(pet != 0,1,0)
        pet += np.random.normal(0, sigma,pet.shape) * non_zero_mask
    return pet*scaler

def segment_BRATS_volume_cell_distribusion(volume_cell_distribution,th_necro,th_up,th_down):
    volume_segmentation = np.zeros_like(volume_cell_distribution)
    volume_segmentation[np.where(volume_cell_distribution >= th_up)] = 1
    volume_segmentation[np.where(np.logical_and(volume_cell_distribution < th_up,volume_cell_distribution >= th_down))] = 3
    volume_segmentation[np.where(volume_cell_distribution >= th_necro)] = 4
    return volume_segmentation

def seg_BRATS_to_volume_SIMPLE(seg,th_necro,th_up,th_down):
    # Define your conditions and corresponding values:
    conditions = [np.isclose(seg, 0.0), 
                  np.isclose(seg, 1.0), 
                  np.isclose(seg, 3.0), 
                  np.isclose(seg, 4.0)]
    values = [0, th_up, th_down, th_necro]

    # Use np.select to apply conditions and create a new array
    new_array = np.select(conditions, values)
    
    # Return the new array
    return new_array

def gibbs_sampler_3d(size, num_iter, noise_std_dev):
    # Initialize random field
    field = np.random.rand(size, size, size)

    # Gibbs sampling
    for it in range(num_iter):
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Compute local mean
                    local_sum = np.sum(field[max(0,i-1):min(size,i+2), 
                                             max(0,j-1):min(size,j+2),
                                             max(0,k-1):min(size,k+2)])
                    local_count = (min(size,i+2)-max(0,i-1))*(min(size,j+2)-max(0,j-1))*(min(size,k+2)-max(0,k-1))
                    local_mean = local_sum/local_count

                    # Draw new value
                    field[i,j,k] = np.random.normal(local_mean, noise_std_dev)

    return field

def m_Tildas(WM,GM,th):
        
    WM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(WM,-1,axis=0) + WM)/2,0)
    WM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(WM,-1,axis=1) + WM)/2,0)
    WM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(WM,-1,axis=2) + WM)/2,0)

    GM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(GM,-1,axis=0) + GM)/2,0)
    GM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(GM,-1,axis=1) + GM)/2,0)
    GM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(GM,-1,axis=2) + GM)/2,0)
    
    return {"WM_t_x": WM_tilda_x,"WM_t_y": WM_tilda_y,"WM_t_z": WM_tilda_z,"GM_t_x": GM_tilda_x,"GM_t_y": GM_tilda_y,"GM_t_z": GM_tilda_z}

def get_D(WM,GM,th,Dw,Dw_ratio):
    M = m_Tildas(WM,GM,th)
    D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
    D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    D_minus_z = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    
    D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
    D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
    D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
    
    return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}

def FK_update(A,D_domain,f, dt,dx,dy,dz):
    D = D_domain
    SP_x = 1/(dx*dx) * (D["D_plus_x"]* (np.roll(A,1,axis=0) - A) - D["D_minus_x"]* (A - np.roll(A,-1,axis=0)) )
    SP_y = 1/(dy*dy) * (D["D_plus_y"]* (np.roll(A,1,axis=1) - A) - D["D_minus_y"]* (A - np.roll(A,-1,axis=1)) )
    SP_z = 1/(dz*dz) * (D["D_plus_z"]* (np.roll(A,1,axis=2) - A) - D["D_minus_z"]* (A - np.roll(A,-1,axis=2)) )
    SP = SP_x + SP_y + SP_z
    diff_A = (SP + f*np.multiply(A,1-A)) * dt
    A += diff_A
    return A

def gauss_sol3d(x,y,z):
    #experimentally chosen
    Dt = 15.0
    M = 1500
    
    gauss = M/np.power(4*np.pi * Dt,3/2) * np.exp(- (np.power(x,2) + np.power(y,2) + np.power(z,2))/(4*Dt))
    gauss = np.where(gauss>0.1, gauss,0)
    gauss = np.where(gauss>1, np.float64(1),gauss)
    return gauss


def get_initial_configuration(Nx,Ny,Nz,NxT,NyT,NzT,r):
    A =  np.zeros([Nx,Ny,Nz])
    if r == 0:
        A[NxT, NyT,NzT] = 1
    else:
        A[NxT-r:NxT+r, NyT-r:NyT+r,NzT-r:NzT+r] = 1
    
    return A

# make the colormaps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['black','white'],256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2',['black','green','yellow','red'],256)

#%%
# update in time
days = 100
# grid size
Nx = sGM.shape[0]
Ny = sGM.shape[1]
Nz = sGM.shape[2]

# grid steps
dx =  1
dy =  1
dz =  1

# Calculate the absolute positions based on percentages
NxT1 = int(args.NxT1_pct * Nx)
NyT1 = int(args.NyT1_pct * Ny)
NzT1 = int(args.NzT1_pct * Nz)

NxT2 = int(args.NxT2_pct * Nx)
NyT2 = int(args.NyT2_pct * Ny)
NzT2 = int(args.NzT2_pct * Nz)

NxT3 = int(args.NxT3_pct * Nx)
NyT3 = int(args.NyT3_pct * Ny)
NzT3 = int(args.NzT3_pct * Nz)


Dw = args.Dw
f = args.rho
r = 1
Nt = days * 10 * np.power((Dw/0.05),1)
dt = days/Nt
print(f'Number of steps: {np.ceil(Nt)}')

N_simulation_steps = int(np.ceil(Nt))
RatioDw_Dg = args.RatioDw_Dg

yv, xv,zv = np.meshgrid(np.arange(0,sGM.shape[1]), np.arange(0,sGM.shape[0]),np.arange(0,sGM.shape[2]))
A = np.array(gauss_sol3d(xv - NxT1 ,yv - NyT1,zv-NzT1))

if NxT2 > 0 and NyT2 > 0 and NzT2 > 0:
    A += np.array(gauss_sol3d(xv - NxT2, yv - NyT2, zv - NzT2))

if NxT3 > 0 and NyT3 > 0 and NzT3 > 0:
    A += np.array(gauss_sol3d(xv - NxT3, yv - NyT3, zv - NzT3))


# Initialize the output file name
output_path = args.out_dir
sGM_nii = nib.Nifti1Image(sGM, affine)
sWM_nii = nib.Nifti1Image(sWM, affine)


nib.save(sGM_nii,f"{output_path}/_gm_",)
nib.save(sWM_nii,f"{output_path}/_wm_",)

output_path_postfix = ''
col_res = np.zeros([2, Nx, Ny, Nz])
Z = 0
try:
    # Simulation code
    D_domain = get_D(sWM, sGM, 0.1, args.Dw, args.RatioDw_Dg)
    
    col_res[0] = copy.deepcopy(A)  # init
        
    # Plot and save initial state
    Z = find_best_z(A)
    plt.figure()
    plt.imshow(sWM[:, :, Z], cmap=cmap1, vmin=0, vmax=1, alpha=1) 
    plt.imshow(col_res[0][:, :, Z], vmin=0, cmap=cmap2, vmax=1, alpha=0.6) 
    plt.savefig(f"{output_path}/init.png")

    for t in range(N_simulation_steps):
        if t % 100 == 0:
            print(t)
        A = FK_update(A, D_domain, args.rho, dt, dx, dy, dz)

    col_res[1] = copy.deepcopy(A)  # final
    print(col_res.shape)
    Z = find_best_z(A)
    # Plot and save final state
    plt.figure()
    plt.imshow(sWM[:, :, Z], cmap=cmap1, vmin=0, vmax=1, alpha=1) 
    plt.imshow(col_res[1][:, :, Z], vmin=0, cmap=cmap2, vmax=1, alpha=0.6)
    plt.savefig(f"{output_path}/final.png")
    

except Exception as e:
    # If there's an error, save what we have with a "FAILED" postfix
    print(f"An error occurred: {e}")
    output_path_postfix = "_FAILED"
finally:
    # Save the result, either the full simulation or whatever was completed if there was an error
    #np.save(f"{output_path}/result{output_path_postfix}", col_res)
    res_nii = nib.Nifti1Image(col_res[-1], affine)
    nib.save(res_nii,f"{output_path}/result{output_path_postfix}")
    print(f"Data saved at {output_path}")
    
################ segmentation
th_necro = args.th_necro
th_up = args.th_up
th_down = args.th_down


segBRATS = segment_BRATS_volume_cell_distribusion(col_res[-1],th_necro,th_up,th_down)
volume3D = seg_BRATS_to_volume_SIMPLE(segBRATS,th_necro,th_up,th_down)

#np.save(f"{output_path}/segm{output_path_postfix}", segBRATS)
plt.figure()
plt.imshow(sWM[:, :, Z], cmap=cmap1, vmin=0, vmax=1, alpha=1) 
plt.imshow(volume3D[:, :, Z], vmin=0, cmap=cmap2, vmax=1, alpha=0.6)
plt.savefig(f"{output_path}/segm.png")

segBRATS_nii = nib.Nifti1Image(segBRATS, affine)
nib.save(segBRATS_nii,f"{output_path}/segm",)

###### pet
data =  col_res[-1]
noise = gibbs_sampler_3d(size=data.shape[0], num_iter=3, noise_std_dev=0.005)
data_noisy = data + noise
data_noisy = data_noisy/np.max(data_noisy)
pet_exact = get_pet_signal(data_noisy,0,0.75,th_necro,th_up,th_down)
undersampled_data = pet_exact[::4, ::4, ::4]
# Compute the scaling factor for each dimension
factors = [data_dim/undersampled_dim for data_dim, undersampled_dim in zip(data.shape, undersampled_data.shape)]
# Restore the resolution using trilinear interpolation
restored_data = np.array(zoom(undersampled_data, factors, order=1)).astype(float)
# Subtract mean from the array
restored_data -= np.mean(restored_data)*0.9

# Add the absolute value of the minimum (if it's negative) to make the smallest value 0
restored_data = np.where(restored_data > 0, restored_data,0)
# Divide by the max to scale to the range [0, 1]
restored_data /= np.max(restored_data)


restored_data = np.where(sWM + sGM > 0.1, restored_data, 0)



plt.figure()
plt.imshow(sWM[:, :, Z], cmap=cmap1, vmin=0, vmax=1, alpha=1) 
plt.imshow(restored_data[:, :, Z], vmin=0, cmap=cmap2, vmax=1, alpha=0.6)
plt.savefig(f"{output_path}/pet.png")
#np.save(f"{output_path}/_pet_{output_path_postfix}", restored_data)
pet_nii = nib.Nifti1Image(restored_data, affine)
nib.save(pet_nii,f"{output_path}/_pet_")
