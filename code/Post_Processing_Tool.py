#################################################################################
####                       POST PROCESSING TOOL                              ####
#################################################################################

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import glob, re, math
import h5py

rc('text', usetex=True)
rc('font', family='serif')

#############################################
##     Organize Folders and other Stuf     ##
#############################################
base_dir = os.path.expanduser("...")
snapshots_dir = os.path.join(base_dir, "Snapshots")
samples_dir = os.path.join(base_dir, "Samples")

#############################################
##          Reference Parameters           ##
#############################################
rho_0  = 1.0                        # Densidad de referencia [kg/m3]
u_tau  = 1.0                        # Velocidad de fricción [m/s]
delta  = 1.0                        # Mitad del canal [m]
Re_tau = 180.0                      # Número de Reynolds de fricción [-]
mu_ref = rho_0 * u_tau * delta / Re_tau   # Viscosidad dinámica [Pa s]
nu_ref = mu_ref / rho_0             # Viscosidad cinemática [m2/s]
tau_w  = rho_0 * u_tau * u_tau      # Tensión de pared de referencia [Pa]

######################################################
##          i) Open Multiple Data Files             ##
###################################################### 
#  3d_turbulent_channel_flow_21350000-DATA.h5
file_indices = list(...)
data_files = [f"3d_turbulent_channel_flow_{i}.h5" for i in file_indices]

with h5py.File(f".../3d_turbulent_channel_flow-MESH.h5", 'r') as data_file:
    x_data = data_file['x'][:,:,:]
    y_data = data_file['y'][:,:,:]
    z_data = data_file['z'][:,:,:]

for file_name in data_files:
    file_path = os.path.join(snapshots_dir, file_name)
    print(f"Loading data: {file_name}")
    
    with h5py.File(file_path, 'r') as data_file:
        u_data = data_file['u'][:,:,:]
        v_data = data_file['v'][:,:,:]
        w_data = data_file['w'][:,:,:]
        P_data = data_file['P'][:,:,:]

        num_points_x  = u_data[0,0,:].size
        num_points_y  = u_data[0,:,0].size
        num_points_z  = u_data[:,0,0].size
        num_points_xz = num_points_x * num_points_z
    
    ######################################################
    ##    ii) Perform the Domain without Ghost Cells    ##
    ######################################################
    # Operating with the Ghost Cells
    # Delete Ghost Cells in P_data
    P_data = np.array(P_data)
    P_data[0, :, :]   = P_data[0, :, :]   + P_data[1, :, :]
    P_data[-1, :, :]  = P_data[-1, :, :]  + P_data[-2, :, :]
    P_data[:, 0, :]   = P_data[:, 0, :]   + P_data[:, 1, :]
    P_data[:, -1, :]  = P_data[:, -1, :]  + P_data[:, -2, :]
    P_data[:, :, 0]   = P_data[:, :, 0]   + P_data[:, :, 1]
    P_data[:, :, -1]  = P_data[:, :, -1]  + P_data[:, :, -2]

    # Delete Ghost Cells in u_input
    u_input = np.array(u_data)
    u_input[0, :, :]   = u_input[0, :, :]   + u_input[1, :, :]
    u_input[-1, :, :]  = u_input[-1, :, :]  + u_input[-2, :, :]
    u_input[:, 0, :]   = 0.
    u_input[:, -1, :]  = 0.
    u_input[:, :, 0]   = u_input[:, :, 0]   + u_input[:, :, 1]
    u_input[:, :, -1]  = u_input[:, :, -1]  + u_input[:, :, -2]

    # Delete Ghost Cells in v_input
    v_input = np.array(v_data)
    v_input[0, :, :]   = v_input[0, :, :]   + v_input[1, :, :]
    v_input[-1, :, :]  = v_input[-1, :, :]  + v_input[-2, :, :]
    v_input[:, 0, :]   = 0.
    v_input[:, -1, :]  = 0.
    v_input[:, :, 0]   = v_input[:, :, 0]   + v_input[:, :, 1]
    v_input[:, :, -1]  = v_input[:, :, -1]  + v_input[:, :, -2]

    # Delete Ghost Cells in w_input
    w_input = np.array(w_data)
    w_input[0, :, :]   = w_input[0, :, :]   + w_input[1, :, :]
    w_input[-1, :, :]  = w_input[-1, :, :]  + w_input[-2, :, :]
    w_input[:, 0, :]   = 0.
    w_input[:, -1, :]  = 0.
    w_input[:, :, 0]   = w_input[:, :, 0]   + w_input[:, :, 1]
    w_input[:, :, -1]  = w_input[:, :, -1]  + w_input[:, :, -2]

    ######################################################
    ##           iii) Compute P_thermo                  ##
    ######################################################
    total_P_volume = 0.0
    total_volume   = 0.0
    for i in range(1, num_points_x-1):
        for j in range(1, num_points_y-1):
            for k in range(1, num_points_z-1):
                # Geometrical stuf
                delta_x = 0.5 * ( x_data[k, j, i+1] - x_data[k, j, i-1] )
                delta_y = 0.5 * ( y_data[k, j+1, i] - y_data[k, j-1, i] )
                delta_z = 0.5 * ( z_data[k+1, j, i] - z_data[k-1, j, i] )
                volume  = delta_x * delta_y * delta_z
                # Update values
                total_P_volume += P_data[k, j, i] * volume
                total_volume   += volume
    P_thermo = total_P_volume / total_volume

    ######################################################
    ##            iv) Obtain P_walls                    ##
    ######################################################
    # Extract pressures at bottom (ymin) and top walls (ymax) 
    P_ymin = P_data[:, 0, :] - P_thermo   
    P_ymax = P_data[:, -1, :] - P_thermo

    ######################################################
    ##         v) Obtain the Shear Stress               ##
    ######################################################
    # Wall shear stress bottom wall
    tau_w_x_bottom = np.zeros( ( num_points_z, num_points_x ) )
    tau_w_z_bottom = np.zeros( ( num_points_z, num_points_x ) )
    # Wall shear stress y = 0
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            # Index of first point in y-direction
            j = 0
            # Streamwise direction
            du_dy = ( u_data[k,j+1,i] - u_data[k,j,i] )/( y_data[k,j+1,i] - y_data[k,j,i] )
            tau_w_x_bottom[k, i] = mu_ref*du_dy
            # Spanwise direction
            dw_dy = ( w_data[k,j+1,i] - w_data[k,j,i] )/( y_data[k,j+1,i] - y_data[k,j,i] )
            tau_w_z_bottom[k, i] = mu_ref*dw_dy

    # Wall shear stress top wall
    tau_w_x_top = np.zeros( ( num_points_z, num_points_x ) )
    tau_w_z_top = np.zeros( ( num_points_z, num_points_x ) )
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            # Index of last point in y-direction
            j = num_points_y-1
            # Streamwise direction
            du_dy = ( u_data[k,j,i] - u_data[k,j-1,i] )/( y_data[k,j,i] - y_data[k,j-1,i] )
            tau_w_x_top[k, i] = mu_ref*du_dy
            # Spanwise direction
            dw_dy = ( w_data[k,j,i] - w_data[k,j-1,i] )/( y_data[k,j,i] - y_data[k,j-1,i] )
            tau_w_z_top[k, i] = mu_ref*dw_dy

    ######################################################
    ##       vi) Construct X_features & Y_features      ##
    ######################################################
    X_features_bottom = torch.stack([
        torch.tensor(P_ymin),
        torch.tensor(tau_w_x_bottom),
        torch.tensor(tau_w_z_bottom)
    ], dim=0)

    X_features_top = torch.stack([
        torch.tensor(P_ymax),
        torch.tensor(tau_w_x_top),
        torch.tensor(tau_w_z_top)
    ], dim=0)

    Y_features = torch.stack([
        torch.tensor(u_input),
        torch.tensor(v_input),
        torch.tensor(w_input)
    ], dim=0)  
    #print(Y_features.shape)

    Y_features_bottom = Y_features[ :, :, 0:65, :]
    #print(Y_features_bottom.shape)
    Y_features_top = torch.flip(Y_features[ :, :, 65:130, :], (2,)) 
    #print(Y_features_top.shape)

    print('X_features_top Shape:',X_features_top.shape)
    print('X_features_bot Shape:',X_features_bottom.shape)
    print('Y_features_top Shape:',Y_features_top.shape)
    print('Y_features_bot Shape:',Y_features_bottom.shape)

    ######################################################
    ##            vii) Data into a H5                   ##
    ######################################################
    out_file_bottom = os.path.join(samples_dir, f"datapost0_{file_name}")
    with h5py.File(out_file_bottom, "w") as f:
        f.create_dataset("X_features", data=X_features_bottom.numpy())
        f.create_dataset("Y_features", data=Y_features_bottom.numpy())
    print(f"Generado: {out_file_bottom}")

    out_file_top = os.path.join(samples_dir, f"datapost1_{file_name}")
    with h5py.File(out_file_top, "w") as f:
        f.create_dataset("X_features", data=X_features_top.numpy())
        f.create_dataset("Y_features", data=Y_features_top.numpy())
    print(f"Generado: {out_file_top}\n")
