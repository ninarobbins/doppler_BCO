import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# import intake
import os 
import yaml
import sys
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from read_zspc_d_qf2 import read_zspc_using_search
from plot_utils import video_from_snapshots

# Config from YAML file
config_path = "../config.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

radar = "MBR2"
date = config["radar_params"][radar]["date"]
file_time=config["radar_params"][radar]["file_time"]
path_to_spectra = config["radar_params"][radar]["path_to_spectra"].format(date=date, file_time=file_time)
path_to_l1 = config["radar_params"][radar]["path_to_l1"].format(date=date, file_time=file_time)
title = config["radar_params"][radar]["plot_title"]
nlevels = config["radar_params"][radar]["reflectivity_levels"]
range_lines = config["radar_params"][radar]["range_indices"]
snapshot_folder = config["radar_params"][radar]["snapshot_folder"].format(date=date, file_time=file_time)
movie_folder = config["radar_params"][radar]["movie_folder"].format(date=date, file_time=file_time)

cbar_kwargs = config["plot_params"]["cbar_kwargs"]
figure_size = config["plot_params"]["figure_size"]
ylim_alt = config["plot_params"]["ylim_alt"]
ylim_spec = config["plot_params"]["ylim_spec"]
vmin_Ze, vmax_Ze = config["plot_params"]["vmin_Ze"], config["plot_params"]["vmax_Ze"]
fs = config["plot_params"]["fontsize"]

start_time = f"{date[:4]}-{date[4:6]}-{date[6:]}T{config['radar_params'][radar]['start_time']}"
end_time = f"{date[:4]}-{date[4:6]}-{date[6:]}T{config['radar_params'][radar]['end_time']}"


# -----------------------------------------------------------
# Folder to save snapshots
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)
    print(f"Folder '{snapshot_folder}' created successfully.")
else:
    print(f"Folder '{snapshot_folder}' already exists.")

# -----------------------------------------------------------
# Folder to save movies
if not os.path.exists(movie_folder):
    os.makedirs(movie_folder)
    print(f"Folder '{movie_folder}' created successfully.")
else:
    print(f"Folder '{movie_folder}' already exists.")


# -----------------------------------------------------------
# Read CORAL NETCDF Spectra

mbr2_spec = read_zspc_using_search(path_to_spectra+f"/{date}_{file_time}.zspc.bz2").sel(time=slice(start_time, end_time))    
time = mbr2_spec.time.values
range_num = mbr2_spec.range.values
altitude = 150 + range_num*31.18
fft_line = mbr2_spec.fftline.values
doppler_vel = fft_line * 10.66145/256 - 10.66145

SPCco = mbr2_spec.power[:,0,:,:].fillna(mbr2_spec.hsdv[:,0,:]).values
SPCco[:,0:99,:] = mbr2_spec.power[:,0,0:99,:].fillna(mbr2_spec.hsdv[:,0,0:99]).values

# -----------------------------------------------------------
# Read CORAL Level 1 data

# cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/catalog.yaml")
# mbr2_ds = cat.BCO.radar_MBR2_c4_v1.to_dask()
mbr2_ds = xr.open_dataset(path_to_l1+f"/MMCR__MBR2__Spectral_Moments__2s__155m-18km__{date[2:]}.nc")
mbr2_ds = mbr2_ds.sel(time=slice(start_time, end_time)) 

print(mbr2_ds)
print(mbr2_spec)

# -----------------------------------------------------------
# Plot radar data evolution in time

# Initialize Reflectivity and Mean Doppler Velocity plots only once
fig = plt.figure(figsize=figure_size, facecolor="white")
plt.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.95, hspace=0.15, wspace=0.05)
fig.suptitle(title + f" - {date}", fontsize=fs)

# Set up grid plot
spec = GridSpec(ncols=5, nrows=6, width_ratios=[0.3, 0.01, 0.05, 0.3, 0.01], height_ratios=[0.5, 0.5, 0.1, 1, 0.1, 0.1], figure=fig)
ax1 = fig.add_subplot(spec[0, :4])
ax2 = fig.add_subplot(spec[1, :4], sharex=ax1)
cax1 = fig.add_subplot(spec[0, 4])
cax2 = fig.add_subplot(spec[1, 4])
ax3 = fig.add_subplot(spec[3, 0])
cax3 = fig.add_subplot(spec[3, 1])
ax4 = fig.add_subplot(spec[3, 3:-1])

# Plot Reflectivity (Static)
Ze_cloud = np.where(mbr2_ds.Ze < -50, np.nan, mbr2_ds.Ze)
levels_Ze = np.linspace(vmin_Ze, vmax_Ze, nlevels + 1)
im_Ze = ax1.contourf(mbr2_ds.time.values.astype("datetime64[ns]"), mbr2_ds.range.values, Ze_cloud.transpose(), cmap="viridis", levels=levels_Ze, zorder=-10)
cbar_Ze = fig.colorbar(im_Ze, cax=cax1, **cbar_kwargs)
cbar_Ze.set_label("$Z_e$ [dBZ]", fontsize=fs, rotation=270, labelpad=15)
cbar_Ze.ax.tick_params(labelsize=fs - 1)
ax1.set_ylim(ylim_alt)
ax1.set_ylabel("Range [m]", fontsize=fs)
ax1.margins(x=0)
ax1.tick_params(axis="both", labelsize=fs - 1)
plt.setp(ax1.get_xticklabels(), visible=False)

# Plot Mean Velocity (Static)
levels_vel = np.linspace(-3, 3, nlevels + 1)
norm = TwoSlopeNorm(vmin=-3, vmax=2, vcenter=0)
im_vel = ax2.contourf(mbr2_ds.time.values.astype("datetime64[ns]"), mbr2_ds.range.values, mbr2_ds.VEL.transpose(), cmap="coolwarm", levels=levels_vel, zorder=-10, norm=norm)
cbar_vel = fig.colorbar(im_vel, cax=cax2, **cbar_kwargs)
cbar_vel.set_label("VEL [ms$^{-1}$]", fontsize=fs, rotation=270, labelpad=15)
cbar_vel.ax.tick_params(labelsize=fs - 1)
ax2.set_ylim(ylim_alt)
ax2.set_ylabel("Range [m]", fontsize=fs)
ax2.margins(x=0)
ax2.tick_params(axis="both", labelsize=fs - 1)
ax2.set_xlabel("Time [UTC]", fontsize=fs)
date_form = DateFormatter("%H:%M")
ax2.xaxis.set_major_formatter(date_form)

# Initialize Spectrum Heat Map and Color Bar
im_spec = ax3.pcolormesh(doppler_vel, altitude, SPCco[0, :, :], cmap="plasma", norm=LogNorm(vmin=ylim_spec[0], vmax=ylim_spec[1]))
cbar_spec = fig.colorbar(im_spec, cax=cax3, **cbar_kwargs)
cbar_spec.set_label("SPCco [power]", fontsize=fs, rotation=270, labelpad=15)
cbar_spec.ax.tick_params(labelsize=fs - 1)
ax3.set_ylim(ylim_alt)
ax3.set_ylabel("Range [m]", fontsize=fs)
ax3.margins(x=0)
ax3.set_xlabel("Doppler Velocity [ms$^{-1}$]", fontsize=fs)

# Initialize vertical lines for the current time step
line_ax1 = ax1.axvline(x=mbr2_ds.time[0].values, linestyle=":", linewidth=2, color="k")
line_ax2 = ax2.axvline(x=mbr2_ds.time[0].values, linestyle=":", linewidth=2, color="k")

# Loop through each time step for Spectrum Heat Map and Lines
for t, timestep in enumerate(mbr2_spec.time.values):

    try:
        # Update vertical line positions for the current time step
        line_ax1.set_xdata([mbr2_ds.time[t].values])
        line_ax2.set_xdata([mbr2_ds.time[t].values])

        # Update Spectrum Heat Map data without recreating color bar
        im_spec.set_array(SPCco[t, :, :].ravel())

        # Clear previous lines in Spectra plot and add new ones
        ax4.clear()
        ax4.plot(doppler_vel, SPCco[t, range_lines[0], :], color="dodgerblue", label=f"{mbr2_ds.range[range_lines[0]].values} m")
        ax4.plot(doppler_vel, SPCco[t, range_lines[1], :], color="red", label=f"{mbr2_ds.range[range_lines[1]].values} m")
        ax4.plot(doppler_vel, SPCco[t, range_lines[2], :], color="orange", label=f"{mbr2_ds.range[range_lines[2]].values} m")

        ax4.set_yscale("log")
        ax4.set_ylim(1e-2, 1e2)
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.tick_right()
        ax4.set_ylabel("SPCco [power]", fontsize=fs, rotation=270, labelpad=15)
        ax4.set_xlabel("Doppler Velocity [ms$^{-1}$]", fontsize=fs)
        ax4.margins(x=0)
        ax4.legend(fontsize=fs, loc="upper left")

        plt.savefig(f"{snapshot_folder}/MBR2_snapshot_{timestep.astype('datetime64[s]').astype(int)}.png", bbox_inches="tight")

    except IndexError:
        break

# -----------------------------------------------------------
# Make movie from snapshots

video_from_snapshots(snapshot_folder, movie_folder + f"/{radar}_{date}_{file_time}.mp4", fps=12)