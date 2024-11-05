import glob
import os 
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from read_zspc_d_qf2 import read_zspc_using_search
from plot_utils import video_from_snapshots

fs = 14


# Read NETCDF spectra
time_band = "0000"
path_to_mbr2 = sorted(glob.glob(f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/data/MBR2/mom/*{time_band}.zspc.bz2"))

for infile in path_to_mbr2:        
    mbr2_spec = read_zspc_using_search(infile)        
    time = mbr2_spec.time.values
    range_num = mbr2_spec.range.values
    altitude = 150 + range_num*31.18
    fft_line = mbr2_spec.fftline.values
    doppler_vel = fft_line * 10.66145/256 - 10.66145

    SPCco = mbr2_spec.power[:,0,:,:].fillna(mbr2_spec.hsdv[:,0,:]).values
    SPCco[:,0:99,:] = mbr2_spec.power[:,0,0:99,:].fillna(mbr2_spec.hsdv[:,0,0:99]).values

# Folder to save snapshots
snapshot_folder = f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/snapshots/heatmaps/MBR2/{path_to_mbr2[0][-22:-9]}"
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)
    print(f"Folder '{snapshot_folder}' created successfully.")
else:
    print(f"Folder '{snapshot_folder}' already exists.")


# Plot spectra
for t,timestep in enumerate(time[:10]):

    print(timestep)

    plt.figure(figsize=(10,6))

    im = plt.pcolormesh(doppler_vel, altitude, SPCco[t,:,:], cmap="plasma", norm=LogNorm(vmin=1e0,vmax=1e4))
    cbar = plt.colorbar(im)
    cbar.set_label("SPCco [power]", rotation=270, labelpad=20, fontsize=fs)
    cbar.ax.tick_params(axis='y', which='major', labelsize=fs-2)
    
    plt.ylim(0,3500)
    plt.xlim(-10,10)
    plt.xlabel("Doppler Velocity [m/s]", fontsize=fs)
    plt.ylabel("Range [m]", fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-2)
    plt.title(f"time={timestep}", fontsize=fs)
    plt.savefig(f"{snapshot_folder}/MBR2_spectra_heatmap_{timestep.astype('datetime64[s]').astype(int)}.png", bbox_inches="tight", dpi=150)
    plt.close()

# Make video from snapshots
movie_folder = f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/movies/heatmaps/MBR2/{path_to_mbr2[0][-22:-9]}"
if not os.path.exists(movie_folder):
    os.makedirs(movie_folder)
    print(f"Folder '{movie_folder}' created successfully.")
else:
    print(f"Folder '{movie_folder}' already exists.")


save_path = f"{movie_folder}/MBR2_spectra_heatmap_{timestep.astype('datetime64[s]').astype(int)}.mp4"
video_from_snapshots(snapshot_folder, save_path, fps=12)