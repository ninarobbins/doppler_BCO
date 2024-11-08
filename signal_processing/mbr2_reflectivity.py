import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os 
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from read_zspc_d_qf2 import read_zspc_using_search
from radar_utils import compute_reflectivity_MBR2

import numpy as np


# -----------------------------------------------------------
# Config from YAML file
config_path = "../config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# -----------------------------------------------------------
# Plotting standards
SIZE = config["plot_params"]["fontsize"]
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

# -----------------------------------------------------------
# Radar info

radar = "MBR2"
date = config["radar_params"][radar]["date"]
file_time=config["radar_params"][radar]["file_time"]
path_to_spectra = config["radar_params"][radar]["path_to_spectra"].format(date=date, file_time=file_time)
path_to_l1 = config["radar_params"][radar]["path_to_l1"].format(date=date, file_time=file_time)
start_time = f"{date[:4]}-{date[4:6]}-{date[6:]}T{config['radar_params'][radar]['start_time']}"
end_time = f"{date[:4]}-{date[4:6]}-{date[6:]}T{config['radar_params'][radar]['end_time']}"
use_time = "2024-09-24T17:40"

# -----------------------------------------------------------
# Read CORAL NETCDF Spectra
mbr2_spec = read_zspc_using_search(path_to_spectra+f"/{date}_{file_time}.zspc.bz2").sel(time=use_time, method="nearest")  
altitude = 150 + mbr2_spec.range.values*31.18
fft_line = mbr2_spec.fftline.values
doppler_vel = fft_line * 10.66145/256 - 10.66145
SPCco = mbr2_spec.power[0, :, :]
RadarConst5 = mbr2_spec.RadarConst5.values
COFA = mbr2_spec.cofa[0, :].values
HSDV = mbr2_spec.hsdv[0, :].values
NPW1 = mbr2_spec.NPW1.values
range_gate = 155.9+np.arange(605.0, dtype = float)*31.18


# -----------------------------------------------------------
# Read CORAL Level 1 data
mbr2_ds = xr.open_dataset(path_to_l1+f"/MMCR__MBR2__Spectral_Moments__2s__155m-18km__{date[2:]}.nc")
mbr2_ds = mbr2_ds.sel(time=use_time, method="nearest")
Zg_from_ds = mbr2_ds.Zg.values
Zf_from_ds = mbr2_ds.Zf.values
Ze_from_ds = mbr2_ds.Ze.values


# -----------------------------------------------------------
# Compute reflectivity from spectrum and radar constant
Z_from_spec = compute_reflectivity_MBR2(SPCco, HSDV, RadarConst5, range_gate, COFA, NPW1)

# -----------------------------------------------------------
# Plot reflectivities and compare
fig, ax1 = plt.subplots(1,1,figsize=(8,6), facecolor="white")
plt.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.95, hspace=0.15, wspace=0.05)

ax1.plot(Z_from_spec, mbr2_ds.range, c="dodgerblue", label="recomputed")
ax1.plot(Zg_from_ds, mbr2_ds.range, c="orange", label=f"Zg {radar} data", zorder=-1)

ax1.set_xlabel("Reflectivity / dBZ")
ax1.set_ylabel("Altitude / m")
ax1.legend()
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines[["left", "bottom"]].set_position(("outward", 5))


plt.savefig(f"/Users/ninarobbins/Desktop/PhD/Rain_Evaporation/figures/reflectivity/{radar}_comparison.png",
            bbox_inches="tight")
plt.show()
