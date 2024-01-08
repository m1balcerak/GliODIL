#!/usr/bin/env python3

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(grandparent_dir)

from collections import defaultdict

import argparse
import json
import math
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
import time
import nibabel as nib
import util_op
from util import printlog, set_log_file
from util import TIME, TIMECLEAR
from util_op import extrap_linear, extrap_quadh, extrap_quad

import linsolver
import matplotlib.pyplot as plt
import util
from scipy import ndimage
import copy
from six.moves import cPickle as pickle #for performance
import math as m

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


g_time_start = time.time()
g_time_callback = 0.
order = 1
offset_D = 0.0
offset_f = 0.0
pet_scaler = 0.8
pet_sigma = 0.0
th_up = 0.55
th_down = 0.1
trim_scale = 1.5
th_matter = 0.1

# Initial values.
global  D_ch, f_ch
D_ch = 1.5
f_ch = 0.12
Dw_ratio_ch = 100
th_up = 0.65
th_down = 0.3
pet_bkg_lvl_ch = 0.3

def read_image_from_nifty_filename_3D(nifty_filename):
    volume_array = nib.load(nifty_filename).get_fdata()
    return volume_array


def get_exact(timesteps, Nx, Ny, Nz, seg_all_path, wm_path, gm_path, pet_path):
    global trim_scale
    
    # choose according to your segmentations
    necrotic = 4
    enhancing = 1
    edema = 3
    
    # read the segmentation, white matter, grey matter, and PET images using the provided full paths
    seg_all = read_image_from_nifty_filename_3D(seg_all_path)
    seg = np.where(seg_all == edema, 1, np.where(np.isin(seg_all, [enhancing, necrotic]), 2, 0))
    print(seg.shape)
    WM_volume = read_image_from_nifty_filename_3D(wm_path)
    GM_volume = read_image_from_nifty_filename_3D(gm_path)
    if args.pet_path != '':
        pet = read_image_from_nifty_filename_3D(pet_path)
        pet = pet / np.max(pet)
        # select region of interest for pet
        pet_select = np.where(np.logical_or(seg_all == edema, seg_all == enhancing),pet,0)
        col_res_trimmed_trimmedspace, sWM_trimmedspace, pet_trimmedspace =  trim4d_space_resolution(np.array([seg]*2),0.1,trim_scale,WM_volume,pet_select)
        pet_lowRes = np.array(ndimage.zoom(pet_trimmedspace,  (Nx/col_res_trimmed_trimmedspace.shape[1], Ny/col_res_trimmed_trimmedspace.shape[2], Nz/col_res_trimmed_trimmedspace.shape[3]),order=0)).clip(min=0)
    else:
        pet_lowRes = None

    col_res_trimmed_trimmedspace, sWM_trimmedspace, sGM_trimmedspace =  trim4d_space_resolution(np.array([seg]*2),0.1,trim_scale,WM_volume,GM_volume)

    
    print(col_res_trimmed_trimmedspace.shape)

    assert col_res_trimmed_trimmedspace[0].shape == sWM_trimmedspace.shape
    assert col_res_trimmed_trimmedspace[0].shape == sGM_trimmedspace.shape
    
    seg_lowRes = np.array(ndimage.zoom(col_res_trimmed_trimmedspace[-1],  (Nx/col_res_trimmed_trimmedspace.shape[1], Ny/col_res_trimmed_trimmedspace.shape[2], Nz/col_res_trimmed_trimmedspace.shape[3]),order=0)).clip(min=0)
    full_shape = WM_volume.shape
    return col_res_trimmed_trimmedspace, sWM_trimmedspace, sGM_trimmedspace, seg,seg_lowRes,pet_lowRes,WM_volume,GM_volume,full_shape

def get_pet_signal(u_exact,sigma,scaler):
    seg = segment_volume_cell_distribusion(u_exact,th_up,th_down)
    pet = np.where(seg == 1,u_exact,0)
    if sigma != 0:
        non_zero_mask = np.where(pet != 0,1,0)
        pet += np.random.normal(0, sigma,pet.shape) * non_zero_mask
    return pet*scaler

def guess_solution(init_state,final_state,N_steps):
    solution = np.zeros((N_steps,)+final_state.shape)
    for it in range(N_steps):
        w_init = (1 - it/(N_steps-1))
        w_final = (it/(N_steps-1))**2
        solution[it] = w_init*init_state + w_final*final_state
    return solution

def segment_volume_cell_distribusion(volume_cell_distribution,th_up,th_down):
    volume_segmentation = np.zeros_like(volume_cell_distribution)
    volume_segmentation[np.where(volume_cell_distribution >= th_up)] = 2
    volume_segmentation[np.where(np.logical_and(volume_cell_distribution < th_up,volume_cell_distribution >= th_down))] = 1
    return volume_segmentation

def segment_volume_cell_distribusion_tf(volume_cell_distribution,th_up_th_down):
    volume_segmentation = tf.zeros_like(volume_cell_distribution)
    volume_segmentation[tf.where(volume_cell_distribution >= th_up)] = 2
    volume_segmentation[tf.where(tf.logical_and(volume_cell_distribution < th_up,volume_cell_distribution >= th_down))] = 1
    return volume_segmentation

def segmentation_to_distribution_bottom(segmentation_volume,th_up,th_down):
    distribution = np.where(segmentation_volume == 2, th_up,0.)
    distribution = np.where(segmentation_volume == 1, th_down,distribution)
    distribution = np.where(segmentation_volume == 0, 0.0,distribution)
    return distribution
def segmentation_to_distribution_ceiling(segmentation_volume,th_up,th_down):
    distribution = np.where(segmentation_volume == 2, 1.0,0.)
    distribution = np.where(segmentation_volume == 1, th_up,distribution)
    distribution = np.where(segmentation_volume == 0, th_down,distribution)
    return distribution

def segmentation_to_distribution_bottom_tf(segmentation_volume,th_up,th_down):
    distribution = tf.where(segmentation_volume == 2, th_up,np.float64(0.))
    distribution = tf.where(segmentation_volume == 1, th_down,distribution)
    distribution = tf.where(segmentation_volume == 0, np.float64(0.),distribution)
    return distribution
def segmentation_to_distribution_ceiling_tf(segmentation_volume,th_up,th_down):
    distribution = tf.where(segmentation_volume == 2, np.float64(1),np.float64(0.))
    distribution = tf.where(segmentation_volume == 1, th_up,distribution)
    distribution = tf.where(segmentation_volume == 0, th_down,distribution)
    return distribution


def gauss_sol3d_tf(x,y,z):
    #experimentally chosen
    Dt = 15.0
    M = 1500
    
    gauss = M/tf.math.pow(4*tf.constant(np.float64(m.pi)) * Dt,3/2) * tf.math.exp(- (tf.math.pow(x,2) + tf.math.pow(y,2) + tf.math.pow(z,2))/(4*Dt))
    gauss = tf.where(gauss>0.1, gauss,0)
    gauss = tf.where(gauss>1, np.float64(1),gauss)
    return gauss

def gauss_sol3d(x,y,z):
    #experimentally chosen
    Dt = 15.0
    M = 1500
    
    gauss = M/np.power(4*np.pi * Dt,3/2) * np.exp(- (np.power(x,2) + np.power(y,2) + np.power(z,2))/(4*Dt))
    gauss = np.where(gauss>0.1, gauss,0)
    gauss = np.where(gauss>1, np.float64(1),gauss)
    return gauss

#%%
def paths_to_combined_seg(path_edema,path_necro):
    #read seg maps
    edema_volume = read_image_from_nifty_filename_3D(path_edema)
    necro_volume = read_image_from_nifty_filename_3D(path_necro)
    res = np.zeros(edema_volume.shape)
    res[np.where(edema_volume > 0.5)] = 1
    res[np.where(necro_volume > 0.5)] = 2
    return res

def trim4d_space_resolution(volume_4d,cell_th,scale,WM,GM):
    volume = volume_4d[-1]
    GoodIndexes = (volume > cell_th)
    #get range for x
    x_min = np.argmax(GoodIndexes.any(2).any(1).astype(int))
    x_max = volume.shape[0] - np.argmax(GoodIndexes.any(2).any(1).astype(int)[::-1])
    #get range for y
    y_min = np.argmax(GoodIndexes.any(2).any(0).astype(int))
    y_max = volume.shape[1] - np.argmax(GoodIndexes.any(2).any(0).astype(int)[::-1])
    #get range for z
    z_min = np.argmax(GoodIndexes.any(1).any(0).astype(int))
    z_max = volume.shape[2] - np.argmax(GoodIndexes.any(1).any(0).astype(int)[::-1])
    
    if scale == 1:
        return volume_4d[:,x_min:x_max,y_min:y_max,z_min:z_max], WM[x_min:x_max,y_min:y_max,z_min:z_max], GM[x_min:x_max,y_min:y_max,z_min:z_max]
    else:
        x_min_new = np.ceil(x_min - ((x_min + x_max)/2 - x_min)*(scale-1)).astype(int)
        x_max_new = np.ceil(x_max + ((x_min + x_max)/2 - x_min)*(scale-1)).astype(int)
        
        y_min_new = np.ceil(y_min - ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
        y_max_new = np.ceil(y_max + ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
        
        y_min_new = np.ceil(y_min - ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
        y_max_new = np.ceil(y_max + ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
        
        z_min_new = np.ceil(z_min - ((z_min + z_max)/2 - z_min)*(scale-1)).astype(int)
        z_max_new = np.ceil(z_max + ((z_min + z_max)/2 - z_min)*(scale-1)).astype(int)
        
        return volume_4d[:,x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new], WM[x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new], GM[x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new]
    

def restore_trim4d_space_resolution(volume_4d,cell_th,scale,sol_final,sol_init):
    volume = volume_4d[-1]
    GoodIndexes = (volume > cell_th)
    #get range for x
    x_min = np.argmax(GoodIndexes.any(2).any(1).astype(int))
    x_max = volume.shape[0] - np.argmax(GoodIndexes.any(2).any(1).astype(int)[::-1])
    #get range for y
    y_min = np.argmax(GoodIndexes.any(2).any(0).astype(int))
    y_max = volume.shape[1] - np.argmax(GoodIndexes.any(2).any(0).astype(int)[::-1])
    #get range for z
    z_min = np.argmax(GoodIndexes.any(1).any(0).astype(int))
    z_max = volume.shape[2] - np.argmax(GoodIndexes.any(1).any(0).astype(int)[::-1])
    
    x_min_new = np.ceil(x_min - ((x_min + x_max)/2 - x_min)*(scale-1)).astype(int)
    x_max_new = np.ceil(x_max + ((x_min + x_max)/2 - x_min)*(scale-1)).astype(int)
    
    y_min_new = np.ceil(y_min - ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
    y_max_new = np.ceil(y_max + ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
    
    y_min_new = np.ceil(y_min - ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
    y_max_new = np.ceil(y_max + ((y_min + y_max)/2 - y_min)*(scale-1)).astype(int)
    
    z_min_new = np.ceil(z_min - ((z_min + z_max)/2 - z_min)*(scale-1)).astype(int)
    z_max_new = np.ceil(z_max + ((z_min + z_max)/2 - z_min)*(scale-1)).astype(int)
    
    volume_4d = np.zeros(volume_4d.shape)
    new = volume_4d[:,x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new]
    sol_hres = np.array(ndimage.zoom(sol_final,  (new.shape[1]/sol_final.shape[0],new.shape[2]/sol_final.shape[1],new.shape[3]/sol_final.shape[2]),order=order))
    volume_4d[-1,x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new] = sol_hres
    
    sol_hres_init = np.array(ndimage.zoom(sol_init,  (new.shape[1]/sol_init.shape[0],new.shape[2]/sol_init.shape[1],new.shape[3]/sol_init.shape[2]),order=order))
    volume_4d[0,x_min_new:x_max_new,y_min_new:y_max_new,z_min_new:z_max_new] = sol_hres_init
    
    return volume_4d[-1],volume_4d[0]

def m_Tildas(WM,GM,th):
        
    WM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(WM,-1,axis=0) + WM)/2,0)
    WM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(WM,-1,axis=1) + WM)/2,0)
    WM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(WM,-1,axis=2) + WM)/2,0)

    GM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(GM,-1,axis=0) + GM)/2,0)
    GM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(GM,-1,axis=1) + GM)/2,0)
    GM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(GM,-1,axis=2) + GM)/2,0)
    
    return {"WM_t_x": WM_tilda_x,"WM_t_y": WM_tilda_y,"WM_t_z": WM_tilda_z,"GM_t_x": GM_tilda_x,"GM_t_y": GM_tilda_y,"GM_t_z": GM_tilda_z}

def get_D(WM,GM,Dw,Dw_ratio):
    th = 0.1
    M = m_Tildas(WM,GM,th)
    D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
    D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    D_minus_z = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    
    D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
    D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
    D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
    
    return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}



def FK_update(A,f,D,dt,dx,dy,dz):
    
    SP_x = 1/(dx*dx) * (D["D_plus_x"]* (np.roll(A,1,axis=0) - A) - D["D_minus_x"]* (A - np.roll(A,-1,axis=0)) )
    SP_y = 1/(dy*dy) * (D["D_plus_y"]* (np.roll(A,1,axis=1) - A) - D["D_minus_y"]* (A - np.roll(A,-1,axis=1)) )
    SP_z = 1/(dz*dz) * (D["D_plus_z"]* (np.roll(A,1,axis=2) - A) - D["D_minus_z"]* (A - np.roll(A,-1,axis=2)) )
    SP = SP_x + SP_y + SP_z
    diff_A = (SP + f*np.multiply(A,1-A)) * dt
    A += diff_A
    return A


def forward_run(A, Dw, f, Nt, days, sWM, sGM,dx,dy,dz,Dw_ratio,on_the_fly=True):

    dt = days/Nt
    # grid size
    Nx = A.shape[0]
    Ny = A.shape[1]
    Nz = A.shape[2]

    init = copy.deepcopy(A)
    updater = copy.deepcopy(A)
    D_domain = get_D(sWM,sGM,Dw,Dw_ratio)

    if on_the_fly:
        for t in range(Nt):
            if t%500 == 0:
                printlog(t)
            updater = FK_update(updater,f,D_domain,dt,dx,dy,dz)
        return np.array([init,updater])
    else:
        col_res = []
        for t in range(Nt):
            if t%50 == 0:
                printlog(t)
                col_res.append(copy.deepcopy(updater))
            updater = FK_update(updater,f,D_domain,dt,dx,dy,dz)
        return np.array(col_res)
    
def forward_run_dice_stopping(A, Dw, f, Nt_max,N_t_odil, days, sWM, sGM,dx,dy,dz,Dw_ratio,th_up,th_down,seg_ref):

    dt = days/Nt_max
    # grid size
    Nx = A.shape[0]
    Ny = A.shape[1]
    Nz = A.shape[2]

    updater = copy.deepcopy(A)
    D_domain = get_D(sWM,sGM,Dw,Dw_ratio)
    

    col_res = np.zeros([Nt_max,Nx,Ny,Nz])
    dice_scores = np.zeros([1])
    new_days = days
    
    volume_core_ref = np.sum(seg_ref == 2)
    volume_edema_ref = np.sum(seg_ref == 1)

    
    for t in range(Nt_max):

        updater = FK_update(updater,f,D_domain,dt,dx,dy,dz)
        col_res[t] = copy.deepcopy(updater)
        t_breaking = Nt_max
        
        if t%100 == 0:
            #segment and check dice score
            seg_updater = segment_volume_cell_distribusion(updater,th_up,th_down)

            score_edema = dice_between_segmented(seg_ref, seg_updater, k=1) 
            score_core = dice_between_segmented(seg_ref, seg_updater, k=2) 
            total_score = score_core * volume_core_ref + score_edema * volume_edema_ref

            dice_scores = np.append(dice_scores,total_score)
            
            if len(dice_scores) > 10 and dice_scores[-3] > dice_scores[-2] and dice_scores[-3] > dice_scores[-1] and total_score > 0: #condition to stop
                t_breaking = t - 200
                new_days  = dt*t_breaking
                break
        
        if t%500 == 0:
            printlog(t)
            printlog(f'Core score:{score_core}, edema score: {score_edema}, total score: {total_score}')
            
        
    if t_breaking < Nt_max:
        col_res = col_res[:t_breaking, :, :, :]
        printlog(f'Break after {new_days}/{days} days.')

    downsampled_table = col_res[::col_res.shape[0] // N_t_odil]
    downsampled_table = downsampled_table[0:N_t_odil]
    return downsampled_table, new_days,score_core, score_edema
    

def discrete_laplacian(S,dx,dy,dz):
    Lzz = (np.roll(S, (0,0,+1), (0,1,2)) -2*S + np.roll(S, (0,0,-1), (0,1,2)))/(dz**2)
    Lyy = (np.roll(S, (0,+1,0), (0,1,2)) -2*S + np.roll(S, (0,-1,0), (0,1,2)))/(dy**2)
    Lxx = (np.roll(S, (+1,0,0), (0,1,2)) -2*S + np.roll(S, (-1,0,0), (0,1,2)))/(dx**2)
    S = Lxx + Lyy + Lzz
    return S

def dice_between_segmented_tf(volume1_tf, volume2, k):
    volume1_tf_mask = tf.where(volume1_tf == k,1,0)
    volume2_mask = np.where(np.array(volume2) == k,1,0).astype(np.int32)
    intersection = tf.math.count_nonzero(volume1_tf_mask * tf.constant(volume2_mask))
    sum =  tf.math.count_nonzero(volume1_tf_mask) + tf.constant(np.int64(np.count_nonzero(volume2_mask)))
    if sum != 0:
        return tf.constant(np.int64(2))*intersection/sum
    else:
        return np.float64(1)

def dice_between_segmented(volume1, volume2, k):
    volume1_mask = np.where(np.array(volume1) == k,1,0)
    volume2_mask = np.where(np.array(volume2) == k,1,0)
    intersection = np.count_nonzero(volume1_mask * volume2_mask)
    sum =  np.count_nonzero(volume1_mask) + np.count_nonzero(volume2_mask)
    if sum != 0:
        return 2*intersection/sum
    else:
        return 1


def operator_fd(mod, ctx):
    global args, domain, exact_uu_lowRes_init, exact_uu_lowRes_final,exact_uu, sWM, sGM, CM_pos, final_u_seg
    global seg_lowRes, pet_lowRes, sWM_lowRes, sGM_lowRes
    dt = ctx.step('t')
    dx = ctx.step('x')
    x = ctx.cell_center('x')
    y = ctx.cell_center('y')
    z = ctx.cell_center('z')
    ones = ctx.field('ones')
    zeros = ctx.field('zeros')
    it = ctx.cell_index('t')
    ix = ctx.cell_index('x')
    nt = ctx.size('t')
    nx = ctx.size('x')
    dy = ctx.step('y')
    iy = ctx.cell_index('y')
    ny = ctx.size('y')
    dz = ctx.step('z')
    iz = ctx.cell_index('z')
    nz = ctx.size('z')
    coeff_net = ctx.neural_net('coeff')



    def stencil_var(key):
        st = [
            ctx.field(key,  0, 0, 0, 0),
            ctx.field(key,  0,-1, 0, 0),
            ctx.field(key,  0, 1, 0, 0),
            ctx.field(key, -1, 0, 0, 0),
            ctx.field(key, -1,-1, 0, 0),
            ctx.field(key, -1, 1, 0, 0),
            
            ctx.field(key,  0, 0,-1, 0),
            ctx.field(key,  0, 0, 1, 0),
            ctx.field(key, -1, 0,-1, 0),
            ctx.field(key, -1, 0, 1, 0),

            ctx.field(key,  0, 0,0,-1),
            ctx.field(key,  0, 0,0, 1),
            ctx.field(key, -1, 0,0,-1),
            ctx.field(key, -1, 0,0, 1)
        ]
        return st

    
    # Discretizes the equation with two extra constants.
    coeff = ctx.neural_net('coeff')()[0]

    D =  tf.abs(coeff[0]) + offset_D
    f =  tf.abs(coeff[1]) + offset_f
    x0 = coeff[2]#CM_pos[1] #53#coeff[2]
    y0 = coeff[3]#M_pos[0] #61#coeff[3]
    z0 = coeff[4]#CM_pos[2] #54#coeff[4]
    s = tf.abs(coeff[5])
    th_up_nn = tf.abs(coeff[6])
    th_down_nn = tf.abs(coeff[7])
    Dw_ratio_nn = tf.math.pow(np.float64(10),tf.abs(coeff[8]))
    pet_bkg_lvl = tf.abs(coeff[9])

    u_st = stencil_var('u')
    u, uxm, uxp, um, umxm, umxp, uym, uyp, umym, umyp, uzm, uzp, umzm, umzp = u_st

    u_t = (u - um) / dt
    
    D_M = get_D(sWM_lowRes,sGM_lowRes,Dw=D,Dw_ratio=Dw_ratio_nn)

    u_xx = (D_M["D_plus_x"]*(uxp - u) - D_M["D_minus_x"]*(u - uxm))  / (dx**2)
    u_yy = (D_M["D_plus_y"]*(uyp - u) - D_M["D_minus_y"]*(u - uym))  / (dy**2)
    u_zz = (D_M["D_plus_z"]*(uzp - u) - D_M["D_minus_z"]*(u - uzm))  / (dz**2)
    
    um_xx = (D_M["D_plus_x"]*(umxp - um) - D_M["D_minus_x"]*(um - umxm))  / (dx**2)
    um_yy = (D_M["D_plus_y"]*(umyp - um) - D_M["D_minus_y"]*(um - umym))  / (dy**2)
    um_zz = (D_M["D_plus_z"]*(umzp - um) - D_M["D_minus_z"]*(um - umzm))  / (dz**2)
    
    
    u_xx = 0.5 * (u_xx + um_xx)
    u_yy = 0.5 * (u_yy + um_yy)
    u_zz = 0.5 * (u_zz + um_zz)
    R = 0.5*f*(tf.math.abs(u)*( 1- u ) + tf.math.abs(um)*( 1- um ))

    #transform init/final to lower res
    fi_u_dist_lowRes_bottom = segmentation_to_distribution_bottom_tf(np.rint(seg_lowRes),th_up_nn,th_down_nn)
    fi_u_dist_lowRes_ceiling = segmentation_to_distribution_ceiling_tf(np.rint(seg_lowRes),th_up_nn,th_down_nn)


    #loss scalers
    pdf_scaler = 2400 #* dim_scaler
    BC_scaler = 4

    param_loss_scaler = 1
    BC_init_gaussian_scaler = 1000
    pet_loss_scaler = 0.00
    if args.pet_path != '':
        pet_loss_scaler = 9.0*5
        #PET #only couple to strong pet signals above 10%
        pet_mask = tf.where(pet_lowRes > 0.1, np.float64(1),np.float64(0))
        #pet_loss = mod.where(it == nt-1, (u-s*pet_lowRes)*pet_mask*tf.math.reduce_sum(pet_lowRes)/tf.math.reduce_sum(pet_lowRes*pet_mask), zeros) 
        pet_loss = mod.where(it == nt-1, (u-(s*(pet_lowRes-pet_bkg_lvl)))*pet_mask, zeros)
    else:
        pet_loss = tf.constant(np.float64(0))
         
    u_outside_loss_scaler = 1
    csf_loss_scaler = 0.01

    param_loss = mod.where(s > 1.02, tf.abs(s-1.02), tf.constant(np.float64(0))) 
    param_loss += mod.where(s < 0.50, tf.abs(s-0.50), tf.constant(np.float64(0)))
    param_loss += mod.where(th_down_nn > 0.50, tf.abs(th_down_nn-0.50), tf.constant(np.float64(0))) 
    param_loss += mod.where(th_down_nn < 0.20, tf.abs(th_down_nn-0.20), tf.constant(np.float64(0)))
    param_loss += mod.where(th_up_nn > 0.85, tf.abs(th_up_nn-0.85), tf.constant(np.float64(0))) 
    param_loss += mod.where(th_up_nn < 0.50, tf.abs(th_up_nn-0.50), tf.constant(np.float64(0)))
    param_loss += mod.where(D > 10.0, tf.abs(D-10.00), tf.constant(np.float64(0)))
    param_loss += mod.where(D < 0.01, tf.abs(D-0.01), tf.constant(np.float64(0)))
    #param_loss += mod.where(f > 1.0, tf.abs(f-1.0), tf.constant(np.float64(0)))
    param_loss += mod.where(f < 0.02, tf.abs(f-0.02), tf.constant(np.float64(0)))
    param_loss += mod.where(Dw_ratio_nn > 1000, tf.abs(Dw_ratio_nn-1000), tf.constant(np.float64(0)))
    param_loss += mod.where(Dw_ratio_nn < 10, tf.abs(Dw_ratio_nn-10), tf.constant(np.float64(0)))
    param_loss += mod.where(pet_bkg_lvl > 0.5, tf.abs(pet_bkg_lvl-0.5), tf.constant(np.float64(0)))



    



    #pdf
    pde_res = u_t - (u_xx + u_yy + u_zz) - R
    pde_res = mod.where(it == 0, zeros, pde_res)
    


    #CSF loss
    matter = np.tile(sWM_lowRes + sGM_lowRes,(nt,1,1,1))
    CSF_mask = np.zeros(u.shape)
    CSF_mask[np.where(matter < th_matter)] = 1
    
    CSF_loss = u*tf.constant(CSF_mask)
    
    matter_mask = np.zeros(u.shape)
    matter_mask[np.where(matter > th_matter)] = 1
    
    #N BC:
    
    
    #BC
    #gaussian origin
    BC_init_gaussian = mod.where(it == 0, u -  gauss_sol3d_tf((x[0,:]-x0),(y[0,:]-y0),(z[0,:]-z0)), zeros)


    #CUT LOW init
    #BC_init_gaussian += mod.where(it==0,tf.math.reduce_sum(tf.where(u < 0.2,u,0)),zeros) 
    ##

    mask1 = tf.where(u < fi_u_dist_lowRes_bottom,np.float64(1),0)
    mask2 = tf.where(u > fi_u_dist_lowRes_ceiling,np.float64(1),0)
    fimp_bottom = mod.where(it == nt - 1, (fi_u_dist_lowRes_bottom - u)*mask1, zeros)
    fimp_ceiling = mod.where(it == nt - 1, (u - fi_u_dist_lowRes_ceiling)*mask2, zeros)
    #fimp_bottom = zeros
    #fimp_ceiling = zeros

    # u < 0, u >1 loss
    u_outside_loss =tf.abs(u - tf.clip_by_value(u,0,1))

    #apply pde weight multiplier for experiments
    pdf_scaler *= args.lambda_pde_multiplier

    res = [param_loss*param_loss_scaler, pde_res*pdf_scaler, BC_init_gaussian*BC_init_gaussian_scaler, fimp_bottom*BC_scaler, fimp_ceiling*BC_scaler,u_outside_loss*u_outside_loss_scaler,pet_loss* pet_loss_scaler,CSF_loss*csf_loss_scaler]
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nt', type=int,  help="Grid size in t")
    parser.add_argument('--Nx', type=int,  help="Grid size in x")
    parser.add_argument('--Ny', type=int,  help="Grid size in y")
    parser.add_argument('--Nz', type=int,  help="Grid size in z")
    parser.add_argument('--days', type=int, default=64)
    parser.add_argument('--save_solution', type=str, default="", help="y to save D,rho,u")
    parser.add_argument('--save_forward', type=str, default="", help="y to save forward run")
    parser.add_argument('--save_forward2', type=str, default="", help="y to save forward run")

    parser.add_argument('--final_print', type=str, default="y", help="y to print coeffs")
    parser.add_argument('--postfix', type=str, default="", help="file name postfix")
    parser.add_argument('--code', type=str, default="", help="file code (additional file prefix, used in some plotting functions to identify patients)")
    parser.add_argument('--lambda_pde_multiplier', type=float, default=1.0, help="pde weight multiplier")

    #paths
    parser.add_argument('--seg_path', type=str, required=True, help='Full path to the BRATS segmentation file')
    parser.add_argument('--wm_path', type=str, required=True, help='Full path to the white matter file')
    parser.add_argument('--gm_path', type=str, required=True, help='Full path to the grey matter file')
    parser.add_argument('--pet_path', type=str, required=False, help='Full path to the PET file')
    parser.add_argument('--initial_guess', type=str)
    parser.add_argument('--outdirectory', type=str, required=True, help='Output directory for the results')


    util.add_arguments(parser)
    linsolver.add_arguments(parser)

    parser.set_defaults(plot_every=1000, report_every=1)
    parser.set_defaults(optimizer='lbfgs')
    parser.set_defaults(linsolver='multigrid')
    parser.set_defaults(beta=1e-5)
    parser.set_defaults(initial_guess=False)
    return parser.parse_args()


@tf.function()
def u_deriv(weights, tt, xx, nt, nx):
    net_u = util_op.NeuralNet(weights, tt, xx)
    return net_u(nt, nx)

def find_best_z(volume):
    scores = [np.sum(volume[:,:, z]) for z in range(volume.shape[2])]
    return np.argmax(scores)



def plot(state, epoch, frame):
    global domain, args, shape, sWM,seg_lowRes,pet_lowRes

    path2s = "us_{:05d}maxRef.png".format(frame)
    path2sm = "us_{:05d}Voxel.png".format(frame)
    path2error = "us_{:05d}error.png".format(frame)
    
    
    uu, = np.array([np.array(domain.state_to_field(name, state)) for name in domain.fieldnames])


    
    slices_it = np.linspace(0, domain.shape[0] - 1, 6, dtype=int)  # type: ignore
    print(f"start plotting epoch: {epoch}")
    Z_selected = find_best_z(exact_uu_lowRes[0])

    sWM_lowRes = np.array(ndimage.zoom(sWM, (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order)).clip(min=0)

    ref = (np.array([seg_lowRes]*args.Nt))
    if pet_lowRes is not None:
        ref[0] = pet_lowRes*2
    util.plot_2d(domain, ref[:,:,:,Z_selected]/2, uu[:,:,:,Z_selected], slices_it, path2s, sWM_lowRes[:,:,Z_selected], umin=0, umax=1,plotsWM=True)
    Z_selected = find_best_z(uu[0])
    util.plot_2d(domain, ref[:,:,:,Z_selected]/2, uu[:,:,:,Z_selected], slices_it, path2sm,  sWM_lowRes[:,:,Z_selected], umin=0, umax=1,plotsWM=True)

    #util.plot_2d(domain, exact_uu_lowRes[:,:,:,Z_selected]/2, np.abs(exact_uu_lowRes[:,:,:,Z_selected] - uu[:,:,:,Z_selected]), slices_it, path2error,  sWM_lowRes[:,:,Z_selected], umin=0.0, umax=0.25,plotsWM=False)

def callback(packed, epoch, dhistory=None, opt=None, loss_grad=None):
    tstart = time.time()

    def callback_update_time():
        nonlocal tstart
        global g_time_callback
        t = time.time()
        g_time_callback += t - tstart
        tstart = t

    global frame, csv, csv_empty, packed_prev, domain
    global history, nhistory
    global g_time_callback, g_time_start
    global exact_uu, exact_uu_lowRes

    report = (epoch % args.report_every == 0)
    calc = (epoch % args.report_every == 0 or epoch % args.history_every == 0
            or epoch < args.history_full
            or (epoch % args.plot_every == 0 and (epoch or args.frames)))
    loss = 0

    if packed_prev is None:
        packed_prev = packed

    if calc:
        state = problem.unpack_state(packed)
        state_prev = problem.unpack_state(packed_prev)
        if loss_grad is not None:
            loss = loss_grad(packed, epoch)[0].numpy()

    packed_prev = np.copy(packed)

    memusage = util.get_memory_usage_kb()

    if report:
        printlog("epoch={:05d}".format(epoch))
    if epoch % args.plot_every == 0 and (epoch or args.frames):
        plot(state, epoch, frame) # type: ignore
        frame += 1

    if report:
        printlog("T last: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_last.counters.items()
        ]))
        printlog("T  all: " + ', '.join([
            '{}={:7.3f}'.format(k, t)
            for k, t in problem.timer_total.counters.items()
        ]))
        printlog("memory: {:} MiB".format(memusage // 1024))

    if calc:
        uu, = np.array([np.array(domain.state_to_field(name, state)) for name in domain.fieldnames]) # type: ignore
        # #transform uu to high res
        # uu = np.repeat(uu,int(192/args.Nx),axis=1)
        # uu = np.repeat(uu,int(192/args.Ny),axis=2)
        # uu = np.repeat(uu,int(192/args.Nz),axis=3)
        uu_prev, = np.array([np.array(domain.state_to_field(name, state_prev)) for name in domain.fieldnames]) # type: ignore
        # #transform uu to high res
        # uu_prev = np.repeat(uu_prev,int(192/args.Nx),axis=1)
        # uu_prev = np.repeat(uu_prev,int(192/args.Ny),axis=2)
        # uu_prev = np.repeat(uu_prev,int(192/args.Nz),axis=3)        

        du = uu - uu_prev 

    if report:
        printlog("u={:.05g} du={:.05g}".format(np.max(np.abs(uu)), # type: ignore
                                               np.max(np.abs(du)))) # type: ignore
        offset = np.zeros_like(state.weights["coeff"][1])
        offset[0] = offset_D
        offset[1] = offset_f
        printlog("coeff={:}".format(np.abs(state.weights["coeff"][1])+offset)) # type: ignore        printlog("coeff={:} {:}".format(state.weights["coeff"][1][0], state.weights["coeff"][1][1])) # type: ignore

        loss_terms = problem.eval_loss_terms(state, epoch)
        printlog("loss_terms={:}".format(loss_terms))

    callback_update_time()

    if report:
        printlog()

    if (epoch % args.history_every == 0
            or epoch < args.history_full) and csv is not None:
        assert calc
        history['epoch'].append(epoch)
        history['du_linf'].append(np.max(abs(du))) # type: ignore
        history['du_l1'].append(np.mean(abs(du))) # type: ignore
        history['du_l2'].append(np.mean((du)**2)**0.5) # type: ignore
        history['t_linsolver'].append(
            problem.timer_last.counters.get('linsolver', 0))
        history['t_grad'].append(
            problem.timer_last.counters.get('eval_grad', 0))
        history['t_sparse_fields'].append(
            problem.timer_last.counters.get('sparse_fields', 0))
        history['t_sparse_weights'].append(
            problem.timer_last.counters.get('sparse_weights', 0))

        # u_seg = segment_volume_cell_distribusion_tf(uu[0]) # type: ignore
        # dice_init_edema = dice_between_segmented(init_u_seg,u_seg,k=1)
        # dice_init_necrosis = dice_between_segmented(init_u_seg,u_seg,k=2)
        # history['dice_init_edema'].append(dice_init_edema) # type: ignore
        # history['dice_init_necrosis'].append(dice_init_necrosis) # type: ignore

        # u_f = segment_volume_cell_distribusion_tf(uu[-1]) # type: ignore
        # dice_f_edema = dice_between_segmented(final_u_seg,u_f,k=1)
        # dice_f_necrosis = dice_between_segmented(final_u_seg,u_f,k=2)
        # history['dice_f_edema'].append(dice_f_edema) # type: ignore
        # history['dice_f_necrosis'].append(dice_f_necrosis) # type: ignore


        history['loss'].append(loss)

        history['memusage'].append(memusage) # type: ignore

        callback_update_time()
        history['tt_linsolver'].append(
            problem.timer_total.counters.get('linsolver', 0))
        history['tt_opt'].append(time.time() - g_time_start - g_time_callback) # type: ignore
        history['tt_callback'].append(g_time_callback)  # type: ignore
        #if opt:
            #history['evals'].append(opt.evals)
        #else:
        #    history['evals'].append(0)

        #if dhistory is not None:
        #    for k, v in dhistory.items():
        #        history[k].append(v)
        nhistory += 1

        keys = list(history)
        if csv_empty:
            csv.write(','.join(keys) + '\n')
            csv_empty = False
        for k in history:
            kref = 'epoch'
            assert len(history[k]) == len(history[kref]), \
                "Wrong history size: {:} of '{}' and {:} of '{}'".format(
                    len(history[k]), k, len(history[kref]), kref)
        row = [history[key][-1] for key in keys]
        line = ','.join(map(str, row))
        csv.write(line + '\n')
        csv.flush()
        callback_update_time()


def main():
    global csv, csv_empty, packed_prev, args, problem, domain, frame
    global history, nhistory
    global exact_uu, exact_uu_lowRes, exact_uu_lowRes_init, exact_uu_lowRes_final, shape
    global init_u_seg, final_u_seg, exact_uu, seg_lowRes, pet_lowRes
    global t_in, x_in
    global t_init, x_init, u_init
    global t_bound, x_bound, u_bound, sWM, sGM, CM_pos, sWM_lowRes, sGM_lowRes
    global D_ch, f_ch

    args = parse_args()
    
    outdir = args.outdirectory + args.code + args.postfix
    
    if args.pet_path != '':
        printlog('Pet in path provided - adding pet to the loss function.')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Change current directory to output directory.
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    set_log_file(open("train.log", 'w'))

    csv = None
    csv_empty = True
    if args.history_every:
        csv = open('train.csv', 'w')
    nhistory = 0
    history = defaultdict(lambda: [0 for _ in range(0)])
    packed_prev = None
    
    # Update arguments.
    args.plot_every *= args.every_factor
    args.history_every *= args.every_factor
    args.report_every *= args.every_factor
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    fieldnames = ['u']
    neuralnets = {
        'coeff': [0, 10],  # "Neural net" with zero inputs and two outputs.
                          # Holds inferred [C_DIFF, C_SRC]
    }
    operator = operator_fd

    exact_uu, sWM, sGM, seg, seg_lowRes, pet_lowRes, WM_volume, GM_volume, full_shape = get_exact(
        1,
        args.Nx,
        args.Ny,
        args.Nz,
        args.seg_path,
        args.wm_path,
        args.gm_path,
        args.pet_path
    )
    
    #update initial guesses for D_ch and f_ch
    volume_core_ref = np.sum(seg_lowRes == 2)
    volume_edema_ref = np.sum(seg_lowRes == 1)
    
    f_ch = 0.12
    D_ch = volume_edema_ref / volume_core_ref if volume_core_ref != 0 else 0.12

    # Clip D_ch if it's larger than 25 times f_ch
    if D_ch > 25 * f_ch:
        D_ch = 25 * f_ch

    
    
    #Evaluate exact solution
    init_u_seg =  exact_uu[-1]*0 #TODO
    final_u_seg =  exact_uu[-1]
    #CM_pos = ndimage.center_of_mass(final_u_seg)
    CM_pos = ndimage.center_of_mass(np.where(final_u_seg == 2, 1,0))
    
    domain = util_op.Domain(ndim=4,
                            shape=(args.Nt, args.Nx, args.Ny, args.Nz),
                            lower=(0, 0, 0 ,0),  # type: ignore
                            upper=(args.days,exact_uu.shape[1], exact_uu.shape[2], exact_uu.shape[3]),  # type: ignore

                            varnames=('t', 'x','y','z'),
                            multigrid=args.multigrid,
                            mg_nlvl=4,
                            mg_cell=True,
                            fieldnames=fieldnames,
                            neuralnets=neuralnets)
    
    printlog('multigrid levels:', *domain.mg_nnw)
    printlog('multigrid fields:', *domain.mg_fieldnames)

    printlog(' '.join(sys.argv))



    problem = util_op.Problem(operator, domain)

    state = util_op.State()



    
    #transform tissue distributions to lower resolution
    sWM_lowRes = np.array(ndimage.zoom(sWM,  (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order)).clip(min=0)
    sGM_lowRes = np.array(ndimage.zoom(sGM,  (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order)).clip(min=0)
    
    yv, xv,zv = np.meshgrid(np.arange(0,exact_uu.shape[2]), np.arange(0,exact_uu.shape[1]),np.arange(0,exact_uu.shape[3]))
    init_u_gauss = np.array(gauss_sol3d(xv - CM_pos[0],yv - CM_pos[1],zv-CM_pos[2]))

    #for PLOTTING
    exact_uu_lowRes = np.array([ndimage.zoom(exact_uu[it],  (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order) for it in range(exact_uu.shape[0])]).clip(min=0)
    exact_uu_lowRes_init = exact_uu_lowRes[0]
    exact_uu_lowRes_final = exact_uu_lowRes[-1]
    

    
    if args.initial_guess == 'forward_character':
       start_time = time.time()


       A = np.array(ndimage.zoom(init_u_gauss,  (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order))
       g_solution_lowRes = forward_run(A,D_ch,f_ch, args.Nt*int(args.Nt/8), args.days, sWM=sWM_lowRes,sGM=sGM_lowRes,dx=domain.step('x'),dy=domain.step('y'),dz=domain.step('z'),Dw_ratio=Dw_ratio_ch,on_the_fly=False)
       g_solution_lowRes = g_solution_lowRes[::int(args.Nt/8)]
       
              
       end_time = time.time()
       elapsed_time = end_time - start_time
       printlog(f"Initial guess finished. It took {elapsed_time} seconds to run it.")
       
       
    param_scaler = 1
    if args.initial_guess == 'forward_character_dice_breaking':
       start_time = time.time()


       A = np.array(ndimage.zoom(init_u_gauss,  (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order))
       
       dx= exact_uu.shape[1]/args.Nx
       dy= exact_uu.shape[2]/args.Ny
       dz= exact_uu.shape[3]/args.Nz
       
       g_solution_lowRes, new_days, score_core, score_edema = forward_run_dice_stopping(A, D_ch, f_ch, args.Nt*int(args.Nt/8),args.Nt, args.days,sWM=sWM_lowRes,sGM=sGM_lowRes,dx=dx,dy=dy,dz=dz, Dw_ratio=Dw_ratio_ch,th_up=th_up,th_down=th_down,seg_ref=seg_lowRes)
       param_scaler = new_days/args.days
              
       end_time = time.time()
       elapsed_time = end_time - start_time
       printlog(f"Initial guess forward_character_dice_breaking finished. It took {elapsed_time} seconds to run it.")
       printlog(f"Dice scores for initial guesses: core {score_core}, edema {score_edema}")

    if args.initial_guess:
        #init NN
        coeff_init = np.array([(D_ch-offset_D)*param_scaler, (f_ch-offset_f)*param_scaler,CM_pos[0],CM_pos[1],CM_pos[2],1,th_up,th_down,np.log10(Dw_ratio_ch),pet_bkg_lvl_ch], dtype=domain.dtype)
        state.weights["coeff"] = [np.empty(shape=(0, 10)), coeff_init]    

        frame = 0
        problem.init_missing(state)
        args.ntrainable = len(problem.pack_state(state))
        domain.add_field_to_state(g_solution_lowRes, 'u', state)
    else:
        #init NN
        coeff_init = np.array([D_ch-offset_D, f_ch-offset_f,CM_pos[0],CM_pos[1],CM_pos[2],1,th_up,th_down,np.log10(Dw_ratio_ch),pet_bkg_lvl_ch], dtype=domain.dtype)
        state.weights["coeff"] = [np.empty(shape=(0, 10)), coeff_init]    

        frame = 0
        problem.init_missing(state)
        args.ntrainable = len(problem.pack_state(state))
    
    
    with open('args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    start_time = time.time()
    if args.optimizer == 'newton':
        util.optimize_newton(args, problem, state, callback)
    else:
        util.optimize_opt(args, args.optimizer, problem, state, callback)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    printlog(f"Optimization with {args.optimizer} took {elapsed_time} seconds to run.")
        
    if args.final_print == "y":
        plot(state, 'f', 9999)
        offset = np.zeros_like(state.weights["coeff"][1])
        offset[0] = offset_D
        offset[1] = offset_f
        printlog("Final coeff={:}".format(np.abs(state.weights["coeff"][1])+offset)) # type: ignore  
        
    if args.save_solution == "y":
        #save nifty final
        res_final,res_init = restore_trim4d_space_resolution(np.array([seg]*2),0.1,trim_scale,np.array(domain.state_to_field('u', state)[-1]),np.array(domain.state_to_field('u', state)[0]))

        nifti_file = nib.Nifti1Image(np.array(res_final), np.eye(4))
        nib.save(nifti_file, f'{args.Nt}_{args.Nx}_{args.Ny}_{args.Nz}_solution.nii')
        
        nifti_file_init = nib.Nifti1Image(np.array(res_init), np.eye(4))
        nib.save(nifti_file_init, f'{args.Nt}_{args.Nx}_{args.Ny}_{args.Nz}_solution_init.nii')
        
        #save coeffs to text
        np.save('coeffs.npy', state.weights["coeff"][1])
        
    if args.save_forward == "odil_res": 
        start_time = time.time()
        
        yv, xv,zv = np.meshgrid(np.arange(0,exact_uu.shape[2]), np.arange(0,exact_uu.shape[1]),np.arange(0,exact_uu.shape[3]))
        gaussian_origin = np.array(gauss_sol3d(xv - state.weights["coeff"][1][2],yv - state.weights["coeff"][1][3],zv-state.weights["coeff"][1][4]))
        gaussian_origin_odil_res = np.array(ndimage.zoom(gaussian_origin,  (args.Nx/exact_uu.shape[1], args.Ny/exact_uu.shape[2], args.Nz/exact_uu.shape[3]),order=order))

        f_run = forward_run(gaussian_origin_odil_res, np.abs(state.weights["coeff"][1][0])+offset_D, np.abs(state.weights["coeff"][1][1])+offset_f, args.Nt*int(args.Nt/8), args.days, sWM=sWM_lowRes,sGM=sGM_lowRes,dx=domain.step('x'),dy=domain.step('y'),dz=domain.step('z'),Dw_ratio=np.power(10,state.weights["coeff"][1][8]))


        f_run_restored,res_init = restore_trim4d_space_resolution(np.array([seg]*2),0.1,trim_scale,f_run[-1],np.array(domain.state_to_field('u', state)[0]))

        
        nifti_file = nib.Nifti1Image(np.array(f_run_restored), np.eye(4))
        nib.save(nifti_file, f'{args.Nt}_{args.Nx}_{args.Ny}_{args.Nz}_solution_forward_restored_odil_res.nii')

        end_time = time.time()
        elapsed_time = end_time - start_time
        printlog(f"The forward run odil_res took {elapsed_time} seconds to run.")

                
    if args.save_forward2 == "full_trim_Gauss": 
        start_time = time.time()
        
        printlog(np.abs(state.weights["coeff"][1][0])+offset_D)
        printlog(np.abs(state.weights["coeff"][1][1])+offset_f)
        
        yv, xv,zv = np.meshgrid(np.arange(0,exact_uu.shape[2]), np.arange(0,exact_uu.shape[1]),np.arange(0,exact_uu.shape[3]))
        gaussian_origin = np.array(gauss_sol3d(xv - state.weights["coeff"][1][2],yv - state.weights["coeff"][1][3],zv-state.weights["coeff"][1][4]))
        
        f_run = forward_run(gaussian_origin, np.abs(state.weights["coeff"][1][0])+offset_D, np.abs(state.weights["coeff"][1][1])+offset_f, args.Nt*int(args.Nt/8), args.days, sWM=sWM,sGM=sGM,dx=1,dy=1,dz=1,Dw_ratio=np.power(10,state.weights["coeff"][1][8]))


        f_run_restored,res_init = restore_trim4d_space_resolution(np.array([seg]*2),0.1,trim_scale,f_run[-1],np.array(domain.state_to_field('u', state)[0]))

        
        nifti_file = nib.Nifti1Image(np.array(f_run_restored), np.eye(4))
        nib.save(nifti_file, f'{args.Nt}_{args.Nx}_{args.Ny}_{args.Nz}_solution_forward_restored_full_trim_GaussForward.nii')

        end_time = time.time()
        elapsed_time = end_time - start_time
        printlog(f"The forward run full_trim_Gauss took {elapsed_time} seconds to run.")
        
        
        
    if args.save_forward2 == "full_trim_GaussTS_npy": 
        start_time = time.time()

        printlog(np.abs(state.weights["coeff"][1][0])+offset_D)
        printlog(np.abs(state.weights["coeff"][1][1])+offset_f)

        yv, xv, zv = np.meshgrid(np.arange(0,exact_uu.shape[2]), np.arange(0,exact_uu.shape[1]),np.arange(0,exact_uu.shape[3]))
        gaussian_origin = np.array(gauss_sol3d(xv - state.weights["coeff"][1][2], yv - state.weights["coeff"][1][3], zv - state.weights["coeff"][1][4]))

        f_run = forward_run(gaussian_origin, np.abs(state.weights["coeff"][1][0])+offset_D, np.abs(state.weights["coeff"][1][1])+offset_f, args.Nt*int(args.Nt/8), args.days, sWM=sWM, sGM=sGM, dx=1, dy=1, dz=1, Dw_ratio=np.power(10, state.weights["coeff"][1][8]),on_the_fly=False)

        # Initialize an empty list to store the restored snapshots
        f_run_restored_full = []

        # Loop through each time step and restore
        i = 0
        for t_step in f_run:
            i = i + 1
            print(f'Step {i}/{f_run.shape[0]}')
            f_run_restored, res_init = restore_trim4d_space_resolution(np.array([seg]*2), 0.1, trim_scale, t_step, np.array(domain.state_to_field('u', state)[0]))
            f_run_restored_full.append(f_run_restored)

        # Convert the list to a NumPy array
        f_run_restored_full = np.array(f_run_restored_full)

        # Save the full 4D restored solution
        #nifti_file = nib.Nifti1Image(f_run_restored_full, np.eye(4))
        np.save(f'{args.Nt}_{args.Nx}_{args.Ny}_{args.Nz}_solution_forward_restored_full4D_trim_GaussForward.npy', f_run_restored_full)

        end_time = time.time()
        elapsed_time = end_time - start_time
        printlog(f"The forward run full_trim_GaussTS took {elapsed_time} seconds to run.")

    with open('done', 'w') as f:
        pass

if __name__ == "__main__": 
# %%

    main()
