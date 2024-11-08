import os 
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
import datetime
import glob 

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from plot_utils import video_from_snapshots

fs = 14

# Path to data
path_to_mbrs = "/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/data/MBRS/mom/20230303_0100.znc"
mbrs_ds = xr.open_dataset(path_to_mbrs, chunks={})
print(mbrs_ds.)

# Folder to save snapshots
snapshot_folder = f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/snapshots/heatmaps/MBRS/{path_to_mbrs[0][-22:-4]}"
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)
    print(f"Folder '{snapshot_folder}' created successfully.")
else:
    print(f"Folder '{snapshot_folder}' already exists.")

# Snapshots
for t,timestep in enumerate(mbrs_ds.time[:50]):

    plt.figure(figsize=(10,6))

    im = plt.pcolormesh(mbrs_ds.doppler, mbrs_ds.range, mbrs_ds.SPCco[t,:,:], cmap="plasma", norm=LogNorm(vmin=1e0,vmax=1e4))
    cbar = plt.colorbar(im)
    cbar.set_label("SPCco [DSP]", rotation=270, labelpad=20, fontsize=fs)
    cbar.ax.tick_params(axis='y', which='major', labelsize=fs-2)
    
    plt.ylim(0,3500)
    plt.xlim(-10,10)
    plt.xlabel("Doppler Velocity [m/s]", fontsize=fs)
    plt.ylabel("Range [m]", fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-2)
    plt.title(f"time={datetime.datetime.fromtimestamp(timestep.values)}", fontsize=fs)
    plt.savefig(f"{snapshot_folder}/MBR2_spectra_heatmap_{timestep.values.astype('datetime64[s]').astype(int)}.png", bbox_inches="tight", dpi=150)
    plt.close()


# Make video from snapshots
movie_folder = f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/movies/heatmaps/MBRS/{path_to_mbrs[0][-22:-9]}"
if not os.path.exists(movie_folder):
    os.makedirs(movie_folder)
    print(f"Folder '{movie_folder}' created successfully.")
else:
    print(f"Folder '{movie_folder}' already exists.")

save_path = f"{movie_folder}/MBRS_spectra_heatmap_{timestep.values.astype('datetime64[s]').astype(int)}.mp4"
video_from_snapshots(snapshot_folder, save_path, fps=12)