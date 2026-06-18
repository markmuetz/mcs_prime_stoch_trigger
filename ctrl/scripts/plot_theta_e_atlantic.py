"""Extract wet-bulb potential temperature from UM pp files, convert to
theta_e via Bolton (1980), and plot the (time, pressure) mean over a
small Atlantic domain (26W–22W, 6N–10N)."""

import sys
from pathlib import Path
from glob import glob

import iris
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from convert_wet_build_to_equivalent_potential_temperature import (
    wet_bulb_to_equivalent_potential_temperature,
)

PP_PATTERN = (
    '/gws/ssde/j25b/mcs_prime/mmuetz/data/UM_sims/u-dg135'
    '/share/cycle/20200901T0000Z/engl/um/em1/englaa_pb???.pp'
)

# Domain: 26W–22W, 6N–10N (longitudes in 0–360 convention).
LON_MIN, LON_MAX = 334.0, 338.0
LAT_MIN, LAT_MAX = 6.0, 10.0

pp_files = sorted(glob(PP_PATTERN))
print(f'Loading wet_bulb_potential_temperature from {len(pp_files)} pp files...')
cubes = iris.load(pp_files, 'wet_bulb_potential_temperature')
cube = cubes.concatenate_cube()
print(f'  concatenated shape: {cube.shape}')

da = xr.DataArray.from_iris(cube)

da_domain = da.sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX))
print(f'  domain subset shape: {da_domain.shape}')

theta_w_mean = da_domain.mean(dim=['latitude', 'longitude'])
print(f'  area-mean shape: {theta_w_mean.shape}')

theta_e = xr.DataArray(
    wet_bulb_to_equivalent_potential_temperature(theta_w_mean.values),
    coords=theta_w_mean.coords,
    dims=theta_w_mean.dims,
    name='equivalent_potential_temperature',
    attrs={'units': 'K'},
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

hours = (theta_e.coords['time'] - theta_e.coords['time'][0]) / np.timedelta64(1, 'h')
pressure = theta_e.coords['pressure'].values

for ax, (da_plot, title, cbar_label) in zip(axes, [
    (theta_w_mean, r'$\theta_w$ (wet-bulb pot. temp.)', 'K'),
    (theta_e, r'$\theta_e$ (equiv. pot. temp.)', 'K'),
]):
    pcm = ax.pcolormesh(
        hours, pressure, da_plot.values.T,
        shading='nearest',
    )
    ax.set_xlabel('Forecast hour')
    ax.set_ylabel('Pressure (hPa)')
    ax.invert_yaxis()
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label=cbar_label, pad=0.02)

fig.suptitle(
    'u-dg135 / em1 / 20200901T0000Z\n'
    f'Atlantic domain mean ({abs(LON_MIN-360):.0f}W–{abs(LON_MAX-360):.0f}W, '
    f'{LAT_MIN:.0f}N–{LAT_MAX:.0f}N)',
    fontsize=12,
)
fig.tight_layout()

outpath = Path(__file__).resolve().parents[2] / 'figs' / 'theta_e_atlantic.png'
fig.savefig(outpath, dpi=150)
print(f'Saved {outpath}')
