"""Extract air temperature and relative humidity from UM pp files, compute
theta_e via metpy, and plot (time, pressure) cross-sections for each
experiment, averaged over a small regional domain."""

from glob import glob

import iris
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import xarray as xr

from remake import Remake, rule

import proj_config as conf

slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'standard', 'mem': 16000}
rmk = Remake(config=dict(slurm=slurm_config), check_outputs='never')

EXPTS = list(conf.EXPT_SIM.keys())

DOMAINS = {
    'atlantic_26W-22W_6N-10N': (334.0, 338.0, 6.0, 10.0),
    'bay_of_bengal_84E-94E_8N-18N': (84.0, 94.0, 8.0, 18.0),
    'philippine_sea_127E-137E_7N-17N': (127.0, 137.0, 7.0, 17.0),
    'indian_ocean_55E-85E_5S-5N': (55.0, 85.0, -5.0, 5.0),
}


def calc_theta_e(T, RH, pressure):
    """Compute equivalent potential temperature from T, RH and pressure.

    Uses metpy: RH + T -> dewpoint, then (pressure, T, dewpoint) -> theta_e.
    """
    T_units = T * units.kelvin
    dewpoint = mpcalc.dewpoint_from_relative_humidity(T_units, RH * units.percent)
    theta_e = mpcalc.equivalent_potential_temperature(
        pressure * units.hPa, T_units, dewpoint,
    )
    return theta_e.magnitude


def _pp_pattern(suite, case):
    return str(conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/em1/englaa_pb???.pp')


@rule(
    matrix={
        'expt': EXPTS,
        'case': conf.CASES[8:9],
        'domain': list(DOMAINS),
    },
    outputs={
        'theta_e': str(conf.PATHS['outdir'] / 'theta_e/{expt}/{case}/{domain}/theta_e.nc'),
    },
    uses={
        'DOMAINS': DOMAINS,
        '_pp_pattern': _pp_pattern,
        'calc_theta_e': calc_theta_e,
    },
)
def extract_theta_e(outputs, expt, case, domain):
    suite = conf.EXPT_SIM[expt]
    pp_files = sorted(glob(_pp_pattern(suite, case)))
    print(f'Loading from {len(pp_files)} pp files ({expt}, {case}, {domain})...')

    da_T = xr.DataArray.from_iris(iris.load(pp_files, 'air_temperature').concatenate_cube())
    da_RH = xr.DataArray.from_iris(iris.load(pp_files, 'relative_humidity').concatenate_cube())

    lon_min, lon_max, lat_min, lat_max = DOMAINS[domain]
    sel = dict(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    T_mean = da_T.sel(**sel).mean(dim=['latitude', 'longitude'])
    RH_mean = da_RH.sel(**sel).mean(dim=['latitude', 'longitude'])

    pressure = T_mean.coords['pressure'].values

    theta_e = xr.DataArray(
        calc_theta_e(T_mean.values, RH_mean.values, pressure[np.newaxis, :]),
        coords=T_mean.coords,
        dims=T_mean.dims,
        name='theta_e',
        attrs={'units': 'K', 'long_name': 'equivalent potential temperature'},
    )
    theta_e.to_dataset().to_netcdf(outputs['theta_e'])


def plot_theta_e_inputs(case, domain):
    return {
        f'theta_e_{expt}': extract_theta_e.outputs['theta_e'].format(expt=expt, case=case, domain=domain)
        for expt in EXPTS
    }


@rule(
    matrix={
        'case': conf.CASES[8:9],
        'domain': list(DOMAINS),
    },
    inputs=plot_theta_e_inputs,
    outputs={
        'fig': str(conf.PATHS['figdir'] / 'theta_e/{case}/{domain}/theta_e.png'),
    },
    depends_on=[extract_theta_e],
    uses={'EXPTS': EXPTS, 'DOMAINS': DOMAINS},
)
def plot_theta_e(inputs, outputs, case, domain):
    datasets = {}
    for expt in EXPTS:
        datasets[expt] = xr.open_dataset(inputs[f'theta_e_{expt}'])['theta_e']

    vmin = min(ds.values.min() for ds in datasets.values())
    vmax = max(ds.values.max() for ds in datasets.values())

    fig, axes = plt.subplots(1, len(EXPTS), figsize=(6 * len(EXPTS), 6),
                             layout='constrained')

    for i, (ax, (expt, da)) in enumerate(zip(axes, datasets.items())):
        da.plot.contourf(
            x='time', y='pressure', levels=20, cmap='turbo',
            vmin=vmin, vmax=vmax, ax=ax,
            add_colorbar=False,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Forecast hour')
        ax.set_ylabel('Pressure (hPa)' if i == 0 else '')
        ax.set_title(expt)
        if i > 0:
            ax.set_yticklabels([])

    lon_min, lon_max, lat_min, lat_max = DOMAINS[domain]
    lon_label = f'{abs(lon_min - 360):.0f}W–{abs(lon_max - 360):.0f}W'
    lat_label = f'{lat_min:.0f}N–{lat_max:.0f}N'

    fig.suptitle(
        f'$\\theta_e$ (equiv. pot. temp.) — {case}\n'
        f'Domain mean ({lon_label}, {lat_label}), em1',
    )
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=axes, label='K', pad=0.02, aspect=40)
    plt.savefig(outputs['fig'], dpi=150)


rmk.rules_from_current_module()
