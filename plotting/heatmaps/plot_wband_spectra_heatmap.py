import os 
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
import datetime
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from plot_utils import video_from_snapshots

fs = 14


path_to_wband = "/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/data/WBAND/230303_020001_P07_ZEN.LV0.NC"
wband_ds = xr.open_dataset(path_to_wband)

# Convert times
start_date = datetime.datetime(2001, 1, 1, 0, 0, 0)
wband_time = np.array([start_date + datetime.timedelta(seconds=int(s)) for s in wband_ds.Time.values])

# Folder to save snapshots
snapshot_folder = f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/snapshots/heatmaps/WBAND/{path_to_wband[-28:-15]}"
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)
    print(f"Folder '{snapshot_folder}' created successfully.")
else:
    print(f"Folder '{snapshot_folder}' already exists.")


# Snapshots
for t,timestep in enumerate(wband_time[:100]):
    print(timestep)

    plt.figure(figsize=(10,6))

    im1 = plt.pcolormesh((wband_ds.C1Vel)* 8.093357/256 - 8.093357, wband_ds.C1Range, wband_ds.C1VSpec[t,:,:], cmap="plasma", norm=LogNorm(vmin=1e-4, vmax=1e-1))
    im2 = plt.pcolormesh((wband_ds.C2Vel)* 8.093357/256 - 8.093357, wband_ds.C2Range, wband_ds.C2VSpec[t,:,:], cmap="plasma", norm=LogNorm(vmin=1e-4, vmax=1e-1))
    im3 = plt.pcolormesh((wband_ds.C3Vel)* 8.093357/256 - 8.093357, wband_ds.C3Range, wband_ds.C3VSpec[t,:,:], cmap="plasma", norm=LogNorm(vmin=1e-4, vmax=1e-1))

    cbar = plt.colorbar(im1)
    cbar.set_label("SPCco - Real [mm$^6$m$^{-3}$]", rotation=270, labelpad=20, fontsize=fs)
    cbar.ax.tick_params(axis='y', which='major', labelsize=fs-2)

    plt.axhline(y=wband_ds.C1Range[0],c="k", linestyle=":", label="range C1")
    plt.axhline(y=wband_ds.C1Range[-1],c="k", linestyle=":")
    plt.axhline(y=wband_ds.C2Range[0],c="dodgerblue", linestyle=":", label="range C2")
    plt.axhline(y=wband_ds.C2Range[-1],c="dodgerblue", linestyle=":")

    plt.ylim(0,3500)
    plt.xlim(-10,10)
    plt.xlabel("Doppler Velocity [m/s]", fontsize=fs)
    plt.ylabel("Range [m]", fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-2)
    plt.title(f"time={timestep}", fontsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig(f"{snapshot_folder}/Wband_spectra_heatmap_{timestep}.png", bbox_inches="tight", dpi=150)
    plt.close()

# Make video from snapshots
save_path = f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/movies/heatmaps/WBAND/{path_to_wband[-28:-15]}.mp4"
video_from_snapshots(snapshot_folder, save_path, fps=12)