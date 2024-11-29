"""This code is all well and good BUT, it's not necessary.
You can download 500 hPa geopotential directly from CDS, which I've done in download_era5.py
"""
raise Exception('Obsolete code')
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from xgcm import Grid

from remake import Remake, Rule

from mcs_prime.era5_calc import ERA5Calc

import proj_config as conf
import utils

Rd = 287.06
g = 9.80665
# As used by IFS: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-SpatialreferencesystemsandEarthmodel
Re = 6371229 # m

slurm_config = {'account': 'short4hr', 'partition': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))


# Originally from /home/users/mmuetz/projects/mcs_prime/ctrl/mcs_env_cond_reviews_remakefiles/calc_era5_ecape.py
class Era5ComputeAlt:
    """Compute geopotential (z) from netcdf data, then compute altitude (alt)

    Code taken from existing code for doing the same for GRIB data"""
    def __init__(self, ds):
        self.ds = ds
        self.e5calc = ERA5Calc('/gws/nopw/j04/mcs_prime/mmuetz/data/ERA5/ERA5_L137_model_levels_table.csv')


    def run(self):
        T = self.ds.t.values
        q = self.ds.q.values
        p = self.e5calc.calc_pressure(self.ds.lnsp.values)
        self.p = p
        zsfc = self.ds.z.values

        # Get levels in ascending order of height (starts at 137)
        levels = self.ds.level.values[::-1]
        # print(levels)

        # 0.609133 = Rv/Rd - 1.
        # virtual T
        Tv = T * (1. + 0.609133 * q) * Rd
        z_h = zsfc

        z = np.zeros_like(p)
        for lev in levels:
            lev_idx = lev - 1
            # print(lev, lev_idx)
            z_h, z_f = self.compute_z_level(lev_idx, p, Tv, z_h)
            z[lev_idx] = z_f

        h = z / g
        alt = Re * h / (Re - h)

        self.z = z
        self.h = h
        self.alt = alt
        return z, h, alt

    def compute_z_level(self, lev_idx, p, Tv, z_h):
        '''Compute z at half & full level for the given level, based on T/q/sp'''
        # compute the pressures (on half-levels)
        # ph_lev, ph_levplusone = get_ph_levs(values, lev)
        ph_lev, ph_levplusone = p[lev_idx - 1], p[lev_idx]

        if lev_idx == 0:
            dlog_p = np.log(ph_levplusone / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the
        # full level
        z_f = z_h + (Tv[lev_idx] * alpha)

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (Tv[lev_idx] * dlog_p)

        return z_h, z_f


class CalcERA5_500hPa_geopotential(Rule):
    rule_matrix = {
        'case': conf.CASES,
    }

    @staticmethod
    def rule_inputs(case):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        inputs = {
            f'era5_{t}_{var}': conf.era5_ml_fmtp(var, t.year, t.month, t.day, t.hour)
            for t in times
            for var in ['z', 'lnsp', 't', 'q']
        }
        return inputs

    @staticmethod
    def rule_outputs(case):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')

        outputs = {
            'output': (
                conf.PATHS['outdir']
                / f'era5_processed/geopotential/{d0}-{dlast}/ecmwf-era5_oper_an_sfc_{d0}-{dlast}.geopotential.nc'
            )
        }
        return outputs

    @staticmethod
    def rule_run(inputs, outputs, case):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)

        geopot = []
        for time in times:
            print(time)
            nc_inputs = [v for k, v in inputs.items() if k.startswith(f'era5_{time}_')]
            e5ds =  xr.open_mfdataset(nc_inputs)
            comp_alt = Era5ComputeAlt(e5ds.isel(time=0))
            comp_alt.run()

            # Create a pressure xarray.DataArray
            da_p = xr.DataArray(
                comp_alt.p / 100, # convert from Pa to hPa
                dims=['level', 'latitude', 'longitude'],
                coords=dict(
                    level=e5ds.q.level,
                    latitude=e5ds.q.latitude,
                    longitude=e5ds.q.longitude,
                ),
                attrs=dict(
                    units='hPa',
                    standard='air_pressure',
                )
            )
            # Create a geopotential xarray.DataArray
            da_h = xr.DataArray(
                comp_alt.h,
                dims=['level', 'latitude', 'longitude'],
                coords=dict(
                    level=e5ds.q.level,
                    latitude=e5ds.q.latitude,
                    longitude=e5ds.q.longitude,
                ),
                attrs=dict(
                    units='m',
                    standard='geopotential_height',
                )
            )

            ds_p = da_p.to_dataset(name='pressure')
            grid = Grid(ds_p, coords=dict(pressure={'center': 'level'}), periodic=False)
            h_plev = grid.transform(
                da_h,
                'pressure',
                np.array([500]),
                target_data=ds_p.pressure,
                method='linear'
            )
            geopot.append(h_plev)
        geopot = xr.concat(geopot, dim='time')
        utils.to_netcdf_tmp_then_copy(geopot, outputs['output'])


class PlotERA5_500hPa_geopotential(Rule):
    rule_matrix = {
        'case': conf.CASES,
    }

    rule_inputs =  CalcERA5_500hPa_geopotential.rule_outputs
    @staticmethod
    def rule_outputs(case):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')

        return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / 'era5geopot' / f'geopot.{d0}.png'}

    @staticmethod
    def rule_run(inputs, outputs, case):
        geopot = xr.open_dataarray(inputs['output']).isel(time=0).load()
        # print(geopot)

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(30, 15), layout='constrained')
        levels = np.arange(460, 624, 4)
        lw = [1 if l % 20 != 0 else 2 for l in levels]
        cs = ax.contour(geopot.longitude, geopot.latitude, geopot.sel(pressure=500) / 10, levels=levels, linewidths=lw, colors='k')
        ax.coastlines(color='grey')

        def fmt(x):
            s = f"{x:.0f}"
            return s

        ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10);
        plt.savefig(outputs['fig'])
