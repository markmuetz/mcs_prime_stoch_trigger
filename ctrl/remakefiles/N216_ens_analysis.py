import pandas as pd
from itertools import product

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndimage
import numpy as np
import scipy.stats
import xarray as xr
import xesmf as xe

from remake.util import sysrun
import utils

from remake import Remake, Rule

import proj_config as conf

# from N216_geopotential_processing import CalcERA5_500hPa_geopotential, PlotERA5_500hPa_geopotential

slurm_config = {'account': 'short4hr', 'partition': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

expt_var = [
    (e, v)
    # for e, v in product(EXPT_SIM, ['precip', 'mcsp_calling_freq'])
    for e, v in product(conf.EXPT_SIM, ['mcsp_calling_freq'])
    if not (e == 'ctrl' and v == 'mcsp_calling_freq')
]
# expt_var.append(('stochMCSP', 'tcwv'))

# STASHCODES:
# m01s05i216: precip (instantaneous)
# m01s05i993: calling freq (1-h mean)
# m01s30i461: TCWV (1-h mean)


class RegridImergToN216(Rule):
    """Regrid IMERG to the same grid as N216 simulations."""

    @staticmethod
    def rule_inputs(case, regrid_method):
        # Just load IMERG on the hour "S??0000".
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)

        def h_to_m(h):
            return h * 60

        inputs = {
            f'imerg_{t}': (
                conf.IMERG_FINAL_30MIN_DIR
                / f'{t.year}/{t.month:02d}/{t.day:02d}/'
                / f'3B-HHR.MS.MRG.3IMERG.{t.year}{t.month:02d}{t.day:02d}-S{t.hour:02d}0000-E{t.hour:02d}2959.{h_to_m(t.hour):04d}.V07B.HDF5.nc4'
            )
            for t in times
        }
        inputs['n216ds'] = (
            conf.SIMDIR
            / 'u-dg135/share/cycle/20200701T0000Z/engl/um/englaa_pa.merged.20200701T0000Z.u-dg135.m01s05i216.nc'
        )

        return inputs

    @staticmethod
    def rule_outputs(case, regrid_method):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'output': (
                conf.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/N216.{regrid_method}.3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    rule_matrix = {
        'case': conf.CASES,
        'regrid_method': ['cons', 'non_cons'],
    }

    @staticmethod
    def rule_run(inputs, outputs, case, regrid_method):
        # MUST be a dataset to add bounds correctly. Bounds needed for conservative regrid.
        n216ds = xr.open_dataset(inputs['n216ds'])
        imerg = xr.open_mfdataset([v for k, v in inputs.items() if k.startswith('imerg')])
        print(imerg)
        print(n216ds)

        if regrid_method == 'cons':
            # Use conservative method.
            # Note, this has a far larger effect than I anticipated.
            # It changes to interp. of the spread-error plots substantially, so that
            # vanillaMCSP is best for no spatial avging.
            imerg_regridder = xe.Regridder(imerg, n216ds, method='conservative', periodic=True)
        else:
            imerg_regridder = xe.Regridder(imerg, n216ds, method='bilinear')
        n216imerg = imerg_regridder(imerg.precipitation)
        utils.to_netcdf_tmp_then_copy(n216imerg, outputs['output'])


class RegridERA5ToN216(Rule):
    """Regrid ERA5 to the same grid as N216 simulations."""

    @staticmethod
    def rule_inputs(case, var):
        # Just load IMERG on the hour "S??0000".
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)

        def h_to_m(h):
            return h * 60

        if var == 'tcwv':
            inputs = {f'era5_{t}': conf.era5_sfc_fmtp(var, t.year, t.month, t.day, t.hour) for t in times}
        elif var == 'z':
            month = case[4:6]
            inputs = {
                'era5_z': (
                    conf.PATHS['datadir']
                    / f'ecmwf-era5/misc/2020/{month}'
                    / f'ecmwf-era5_oper_an_pl.2024.{month}.1-11.geopotential_500hPa.nc'
                ),
            }
        inputs['n216ds'] = (
            conf.SIMDIR
            / 'u-dg135/share/cycle/20200701T0000Z/engl/um/englaa_pa.merged.20200701T0000Z.u-dg135.m01s05i216.nc'
        )

        return inputs

    @staticmethod
    def rule_outputs(case, var):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        if var == 'tcwv':
            outputs = {
                'output': (
                    conf.PATHS['outdir']
                    / f'era5_processed/N216grid/{d0}-{dlast}/ecmwf-era5_oper_an_sfc_{d0}-{dlast}.{var}.nc'
                )
            }
        elif var == 'z':
            outputs = {
                'output': (
                    conf.PATHS['outdir']
                    / f'era5_processed/N216grid/{d0}-{dlast}/ecmwf-era5_oper_an_pl_{d0}-{dlast}.{var}.nc'
                )
            }
        return outputs

    rule_matrix = {
        'case': conf.CASES,
        'var': ['tcwv', 'z'],
    }

    @staticmethod
    def rule_run(inputs, outputs, case, var):
        # MUST be a dataset to add bounds correctly. Bounds needed for conservative regrid.
        n216ds = xr.open_dataset(inputs['n216ds'])
        era5ds = xr.open_mfdataset([v for k, v in inputs.items() if k.startswith('era5')])
        print(era5ds)
        print(n216ds)
        print('adding bounds to era5ds and n216ds')
        era5ds_bounds = era5ds.cf.add_bounds(['latitude', 'longitude'])
        # n216ds_bounds = n216ds.cf.add_bounds(['latitude', 'longitude'])

        # Used conservative method.
        e5regridder = xe.Regridder(era5ds_bounds, n216ds, method='conservative', periodic=True)
        n216era5da = e5regridder(era5ds[var])
        utils.to_netcdf_tmp_then_copy(n216era5da, outputs['output'])


class CalcTotalPrecip(Rule):
    @staticmethod
    def rule_inputs(case, regrid_method, ens, red_cf_expt):
        inputs = {}
        for expt in conf.EXPT_SIM:
            suite = conf.EXPT_SIM[expt]
            inputs[f'pflux_{expt}'] = (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/' f'englaa_pa.merged.{case}.{suite}.m01s05i216.nc'
            )
            # inputs[f'pflux_{expt}'] = conf.SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'
        if red_cf_expt:
            suite = 'u-dj618'
            inputs[f'pflux_red_cf'] = (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/' f'englaa_pa.merged.{case}.{suite}.m01s05i216.nc'
            )
        inputs['imerg'] = RegridImergToN216.rule_outputs(case, regrid_method)['output']
        return inputs

    @staticmethod
    def rule_outputs(case, regrid_method, ens, red_cf_expt):
        if red_cf_expt:
            return {
                'total_precip_data': conf.PATHS['outdir']
                / 'N216sims'
                / case
                / f'total_precip.{case}.{regrid_method}.ens_{ens}.inc_red_cf.nc'
            }
        else:
            return {
                'total_precip_data': conf.PATHS['outdir']
                / 'N216sims'
                / case
                / f'total_precip.{case}.{regrid_method}.ens_{ens}.nc'
            }

    rule_matrix = {
        ('case', 'regrid_method', 'ens', 'red_cf_expt'):
        # list(product(conf.CASES, ['cons', 'non_cons'], ['full', 'red'], [False])) +
        [('20200701T0000Z', 'cons', 'red', True)]
    }

    @staticmethod
    def rule_run(inputs, outputs, case, regrid_method, ens, red_cf_expt):
        ds = xr.Dataset()
        if red_cf_expt:
            expts = list(conf.EXPT_SIM) + ['red_cf']
        else:
            expts = conf.EXPT_SIM

        imerg = xr.load_dataarray(inputs['imerg'])
        ds['imerg_ts'] = imerg.mean(dim=['latitude', 'longitude'])
        # IMERG data has a cftimeindex (julian), whereas pflux has a pandas datatimeindex.
        # Convert OR all of the pflux data will be nans.
        # This behaviour has changed - perhaps in an update to xarray?
        ds['time'] = ds.indexes['time'].to_datetimeindex()

        for expt in expts:
            print(expt)
            pflux = xr.load_dataset(inputs[f'pflux_{expt}']).precipitation_flux
            pflux.values *= 3600
            pflux.attrs['units'] = 'mm h-1'
            if ens == 'red':
                pflux = pflux.isel(realization=slice(1, 10))
            print(pflux)
            ds[f'{expt}_ts'] = pflux.mean(dim=['realization', 'latitude', 'longitude'])

        utils.to_netcdf_tmp_then_copy(ds, outputs['total_precip_data'])


class PlotTotalPrecip(Rule):
    rule_inputs = CalcTotalPrecip.rule_outputs

    @staticmethod
    def rule_outputs(case, regrid_method, ens, red_cf_expt):
        if red_cf_expt:
            return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / f'total_precip.{case}.{regrid_method}.ens_{ens}.inc_red_cf.png'}
        else:
            return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / f'total_precip.{case}.{regrid_method}.ens_{ens}.png'}

    rule_matrix = {
        ('case', 'regrid_method', 'ens', 'red_cf_expt'):
        # list(product(conf.CASES, ['cons', 'non_cons'], ['full', 'red'], [False])) +
        [('20200701T0000Z', 'cons', 'red', True)]
    }

    @staticmethod
    def rule_run(inputs, outputs, case, regrid_method, ens, red_cf_expt):
        ds = xr.load_dataset(inputs['total_precip_data'])
        # print(ds)
        imerg_ts = ds['imerg_ts']
        print(imerg_ts)

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(range(len(imerg_ts.time)), imerg_ts, c='k', label='IMERG')
        if red_cf_expt:
            expts = list(conf.EXPT_SIM) + ['red_cf']
        else:
            expts = conf.EXPT_SIM

        for expt in expts:
            pflux_ts = ds[f'{expt}_ts']
            # print(pflux_ts)
            (l,) = ax.plot(range(len(pflux_ts.time)), pflux_ts, label=expt)
            ax2.plot(
                range(len(pflux_ts.time)), pflux_ts.values / imerg_ts.values * 100, c=l.get_color(), ls='--', label=expt
            )
        ax.set_xlabel('Days since start')
        ax.set_ylabel('non-area-weighted global mean precip (mm h$^{-1}$)')
        ax2.set_ylabel('% IMERG')
        ax.legend(loc='center right')
        ntime = len(imerg_ts)
        times_half_days_hours = np.arange(0, ntime + 1, 12)
        times_days = [int(v) for v in times_half_days_hours / 24]
        times_days[1::2] = [''] * len(times_days[1::2])
        ax.set_xticks(times_half_days_hours, times_days)
        ax.set_xlim((0, ntime))
        ax.set_ylim((0, None))
        ax2.set_ylim((90, 200))
        ax2.set_yticks(np.arange(90, 141, 10))
        ax2.axhline(y=100, ls='--', lw=1, c='k')

        plt.savefig(outputs['fig'])


nens = 10
aspect = 1.43  # mean aspect ratio of cells over the tropics.
sel_tropics = {'latitude': slice(-30, 30)}
sigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class GuassianFilterN216Imerg(Rule):
    @staticmethod
    def rule_inputs(case, regrid_method):
        inputs = {'imerg': RegridImergToN216.rule_outputs(case, regrid_method)['output']}
        return inputs

    @staticmethod
    def rule_outputs(case, regrid_method):
        args, kwargs = conf.UM_TIMES[case]
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'imerg_filtered': (
                conf.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/N216.{regrid_method}.guassian_filtered_3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    rule_matrix = {
        'case': conf.CASES,
        'regrid_method': ['cons', 'non_cons'],
    }

    @staticmethod
    def rule_run(inputs, outputs, case, regrid_method):
        imerg = xr.load_dataarray(inputs['imerg'])
        imerg_tropics = imerg.sel(**sel_tropics)
        imerg_filtered = []

        for sigma in sigmas:
            print(sigma)
            imerg_filtered_values = ndimage.gaussian_filter(imerg.sel(**sel_tropics), (0, aspect * sigma, sigma))
            da_imerg_filtered = imerg_tropics.copy()
            da_imerg_filtered.values = imerg_filtered_values
            imerg_filtered.append(da_imerg_filtered)
        imerg_filtered = xr.concat(imerg_filtered, dim=pd.Index(sigmas, name='sigma'))
        utils.to_netcdf_tmp_then_copy(imerg_filtered, outputs['imerg_filtered'])


class GuassianFilterExpt(Rule):
    @staticmethod
    def rule_inputs(expt, case):
        suite = conf.EXPT_SIM[expt]
        inputs = {
            'pflux': (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/' f'englaa_pa.merged.{case}.{suite}.m01s05i216.nc'
            )
        }
        return inputs

    @staticmethod
    def rule_outputs(expt, case):
        suite = conf.EXPT_SIM[expt]
        outputs = {'pflux_filtered': conf.SIMDIR / f'{suite}/processed/{expt}/{case}/engla_pa.filtered_precip.nc'}
        return outputs

    rule_matrix = {
        'expt': list(conf.EXPT_SIM.keys()),
        'case': conf.CASES,
    }

    @staticmethod
    def rule_run(inputs, outputs, expt, case):
        pflux = xr.open_dataset(inputs['pflux']).precipitation_flux.load()
        pflux.values *= 3600
        pflux.attrs['units'] = 'mm h-1'
        pflux_tropics = pflux.sel(**sel_tropics)

        pflux_filtered = []
        for sigma in sigmas:
            da_pflux_filtered = pflux_tropics.copy()

            print(sigma)
            for i in range(nens):
                if sigma != 0:
                    da_pflux_filtered.values[i] = ndimage.gaussian_filter(
                        da_pflux_filtered.values[i], (0, aspect * sigma, sigma)
                    )
            pflux_filtered.append(da_pflux_filtered)
        pflux_filtered = xr.concat(pflux_filtered, dim=pd.Index(sigmas, name='sigma'))
        utils.to_netcdf_tmp_then_copy(pflux_filtered, outputs['pflux_filtered'])


def rmse(a1, a2):
    return np.sqrt(np.mean((a1 - a2) ** 2))


class Calc_eRMSE(Rule):
    @staticmethod
    def rule_inputs(expt, case, regrid_method):
        inputs = {
            'pflux_filtered': GuassianFilterExpt.rule_outputs(expt, case)['pflux_filtered'],
            'imerg_filtered': GuassianFilterN216Imerg.rule_outputs(case, regrid_method)['imerg_filtered'],
        }
        return inputs

    @staticmethod
    def rule_outputs(expt, case, regrid_method):
        suite = conf.EXPT_SIM[expt]
        outputs = {
            'eRMSE': conf.SIMDIR / f'{suite}/processed/{expt}/{case}/engla_pa.filtered_precip.eRMSE.{regrid_method}.nc'
        }
        return outputs

    rule_matrix = {
        'expt': list(conf.EXPT_SIM.keys()),
        'case': conf.CASES,
        'regrid_method': ['cons', 'non_cons'],
    }

    @staticmethod
    def rule_run(inputs, outputs, expt, case, regrid_method):
        pflux_filtered = xr.load_dataarray(inputs['pflux_filtered'])
        imerg_filtered = xr.load_dataarray(inputs['imerg_filtered'])
        ntime = len(pflux_filtered.time)
        eRMSE_data = np.full((len(sigmas), nens, ntime), np.nan)

        for ens_idx in range(nens):
            for sigma_idx in range(len(sigmas)):
                print(expt, ens_idx, sigma_idx)
                for time_idx in range(ntime):
                    eRMSE_data[sigma_idx, ens_idx, time_idx] = rmse(
                        imerg_filtered[sigma_idx, time_idx], pflux_filtered[sigma_idx, ens_idx, time_idx]
                    )

        eRMSE = xr.DataArray(
            eRMSE_data,
            coords={
                'sigma': pflux_filtered['sigma'],
                'realization': pflux_filtered['realization'],
                'time': pflux_filtered['time'],
            },
        )
        utils.to_netcdf_tmp_then_copy(eRMSE, outputs['eRMSE'])


class Calc_dRMSE(Rule):
    @staticmethod
    def rule_inputs(expt, case):
        inputs = {
            'pflux_filtered': GuassianFilterExpt.rule_outputs(expt, case)['pflux_filtered'],
        }
        return inputs

    @staticmethod
    def rule_outputs(expt, case):
        suite = conf.EXPT_SIM[expt]
        outputs = {'dRMSE': conf.SIMDIR / f'{suite}/processed/{expt}/{case}/engla_pa.filtered_precip.dRMSE.nc'}
        return outputs

    rule_matrix = {
        'expt': list(conf.EXPT_SIM.keys()),
        'case': conf.CASES,
    }

    @staticmethod
    def rule_run(inputs, outputs, expt, case):
        pflux_filtered = xr.load_dataarray(inputs['pflux_filtered'])
        ntime = len(pflux_filtered.time)
        dRMSE_data = np.full((len(sigmas), nens, nens, ntime), np.nan)

        for ens_idx1 in range(nens):
            for ens_idx2 in range(ens_idx1 + 1, nens):
                print(expt, ens_idx1, ens_idx2)
                for sigma_idx in range(len(sigmas)):
                    for time_idx in range(ntime):
                        dRMSE_data[sigma_idx, ens_idx1, ens_idx2, time_idx] = rmse(
                            pflux_filtered[sigma_idx, ens_idx1, time_idx], pflux_filtered[sigma_idx, ens_idx2, time_idx]
                        )
        dRMSE = xr.DataArray(
            dRMSE_data,
            coords={
                'sigma': pflux_filtered['sigma'],
                'realization1': pflux_filtered['realization'].values,
                'realization2': pflux_filtered['realization'].values,
                'time': pflux_filtered['time'],
            },
        )
        utils.to_netcdf_tmp_then_copy(dRMSE, outputs['dRMSE'])


def plot_summary_spread_error_ts(expt_dRMSE, expt_eRMSE, ens='full'):
    fig, ax = plt.subplots(1, 1, layout='constrained')
    fig.set_size_inches(8, 6)
    ntime = len(expt_eRMSE['ctrl'].time)

    for i, expt in enumerate(conf.EXPT_SIM):
        dRMSE = expt_dRMSE[expt]
        if ens == 'red':
            dRMSE = dRMSE.sel(realization1=slice(1, 10), realization2=slice(1, 10))
            expt_eRMSE[expt] = expt_eRMSE[expt].sel(realization=slice(1, 10))
        dRMSE_sigma = dRMSE.mean(dim=['realization1', 'realization2', 'time']).values
        eRMSE_sigma = expt_eRMSE[expt].mean(dim=['realization', 'time']).values
        ax.plot(dRMSE.sigma, dRMSE_sigma - eRMSE_sigma, label=f'{expt}')
    ax.axhline(y=0, ls='--', color='k')
    ax.set_xlabel('sigma')
    ax.set_ylabel('spread - error (mm h$^{-1}$)')
    ax.legend()


def plot_spread_error_ts(expt_dRMSE, expt_eRMSE, smooth=False, xlim='full', show_error_minus_spread=False, ens='full'):
    plot_sigmas = [0, 2, 4]
    ntime = len(expt_eRMSE['ctrl'].time)

    fig, axes = plt.subplots(1, len(plot_sigmas), sharex=True, layout='constrained')
    fig.set_size_inches(20, 6)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, sigma in zip(axes, plot_sigmas):
        if smooth:
            ax.set_title(rf'$\sigma=${sigma} ({smooth} h smoothing)')
        else:
            ax.set_title(rf'$\sigma=${sigma}')
        for i, expt in enumerate(conf.EXPT_SIM):
            # dRMSE_ts = np.nanmean(expt_dRMSE[expt], axis=(0, 1))
            # eRMSE_ts = np.nanmean(expt_eRMSE[expt], axis=0)
            dRMSE = expt_dRMSE[expt]
            if ens == 'red':
                dRMSE = dRMSE.sel(realization1=slice(1, 10), realization2=slice(1, 10))
                expt_eRMSE[expt] = expt_eRMSE[expt].sel(realization=slice(1, 10))
            dRMSE_ts = dRMSE.mean(dim=['realization1', 'realization2']).sel(sigma=sigma).values
            eRMSE_ts = expt_eRMSE[expt].mean(dim=['realization']).sel(sigma=sigma).values
            if smooth:
                dRMSE_ts = np.convolve(dRMSE_ts, np.ones((smooth,)) / smooth, mode='same')
                eRMSE_ts = np.convolve(eRMSE_ts, np.ones((smooth,)) / smooth, mode='same')
            c = cs[i]
            ax.plot(eRMSE_ts, color=c, label=f'{expt} error')
            ax.plot(dRMSE_ts, ls='--', color=c, label=f'{expt} spread')
            if show_error_minus_spread:
                ax.plot(dRMSE_ts - eRMSE_ts, ls=':', color=c, label=f'{expt} spread - error')
                ax.axhline(y=0, ls='-', lw=0.5, color='k')

        times_half_days_hours = np.arange(0, ntime + 1, 12)
        times_days = [int(v) for v in times_half_days_hours / 24]
        times_days[1::2] = [''] * len(times_days[1::2])
        ax.set_xticks(times_half_days_hours, times_days)
        ax.set_xlim((0, ntime))
    axes[0].set_ylabel('RMSE (mm h$^{-1}$)')
    if xlim != 'full':
        ax.set_xlim(xlim)
    if smooth and xlim == 'full':
        ax.set_xlim((smooth, 240 - smooth))
    for ax in axes:
        ax.relim()
        if not show_error_minus_spread:
            ax.set_ylim((0, None))
    if len(axes) % 2 == 1:
        midax = axes[len(axes) // 2]
        midax.legend(ncol=len(conf.EXPT_SIM))
        midax.set_xlabel('time (day)')
        ylim = midax.get_ylim()
        midax.set_ylim((ylim[0], ylim[1] * 1.2))
    else:
        axes[-1].legend(ncol=len(conf.EXPT_SIM))
        for ax in axes.flatten():
            ax.set_xlabel('time (day)')


class PlotSpreadError(Rule):
    @staticmethod
    def rule_inputs(plot_kwargs, case, regrid_method):
        inputs = {
            f'{expt}_eRMSE': Calc_eRMSE.rule_outputs(expt, case, regrid_method)['eRMSE'] for expt in conf.EXPT_SIM
        }
        inputs.update({f'{expt}_dRMSE': Calc_dRMSE.rule_outputs(expt, case)['dRMSE'] for expt in conf.EXPT_SIM})
        return inputs

    @staticmethod
    def rule_outputs(plot_kwargs, case, regrid_method):
        kwstr = '-'.join(f'{k}={v}' for k, v in plot_kwargs.items())
        kwstr = kwstr.replace(' ', '')
        return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / f'spread_error.{case}.{kwstr}.{regrid_method}.png'}

    rule_matrix = {
        'plot_kwargs': [
            dict(smooth=False, show_error_minus_spread=True),
            dict(smooth=24),
            dict(xlim=(0, 20)),
            dict(xlim=(0, 48)),
            dict(smooth=False, show_error_minus_spread=True, ens='full'),
            dict(smooth=False, show_error_minus_spread=True, ens='red'),
        ],
        'case': conf.CASES,
        'regrid_method': ['cons', 'non_cons'],
    }

    @staticmethod
    def rule_run(inputs, outputs, plot_kwargs, case, regrid_method):
        print(plot_kwargs, case, regrid_method)

        expt_eRMSE = {expt: xr.load_dataarray(inputs[f'{expt}_eRMSE']) for expt in conf.EXPT_SIM}
        expt_dRMSE = {expt: xr.load_dataarray(inputs[f'{expt}_dRMSE']) for expt in conf.EXPT_SIM}
        print(expt_eRMSE)
        print(expt_dRMSE)
        plot_spread_error_ts(expt_dRMSE, expt_eRMSE, **plot_kwargs)
        plt.savefig(outputs['fig'])


class PlotAllCasesSpreadError(Rule):
    rule_matrix = {
        'plot_kwargs': [
            dict(smooth=False, show_error_minus_spread=True),
            dict(smooth=24),
            dict(xlim=(0, 20)),
            dict(xlim=(0, 48)),
            dict(smooth=False, show_error_minus_spread=True, ens='full'),
            dict(smooth=False, show_error_minus_spread=True, ens='red'),
        ],
        'regrid_method': ['cons', 'non_cons'],
    }

    @staticmethod
    def rule_inputs(plot_kwargs, regrid_method):
        inputs = {
            f'{case}_{expt}_eRMSE': Calc_eRMSE.rule_outputs(expt, case, regrid_method)['eRMSE']
            for expt in conf.EXPT_SIM
            for case in conf.CASES
        }
        inputs.update(
            {
                f'{case}_{expt}_dRMSE': Calc_dRMSE.rule_outputs(expt, case)['dRMSE']
                for expt in conf.EXPT_SIM
                for case in conf.CASES
            }
        )
        return inputs

    @staticmethod
    def rule_outputs(plot_kwargs, regrid_method):
        kwstr = '-'.join(f'{k}={v}' for k, v in plot_kwargs.items())
        kwstr = kwstr.replace(' ', '')
        return {
            'fig': (conf.PATHS['figdir']
            / 'N216sims'
            / 'all_cases'
            / f'spread_error.all_cases.{kwstr}.{regrid_method}.png'),
            'summary_fig': (conf.PATHS['figdir']
            / 'N216sims'
            / 'all_cases'
            / f'spread_error.summary.all_cases.{kwstr}.{regrid_method}.png'),
        }

    @staticmethod
    def rule_run(inputs, outputs, plot_kwargs, regrid_method):
        print(plot_kwargs, regrid_method)
        # expt_eRMSE = {expt: xr.load_dataarray(inputs[f'{expt}_eRMSE']) for expt in conf.EXPT_SIM}
        # expt_dRMSE = {expt: xr.load_dataarray(inputs[f'{expt}_dRMSE']) for expt in conf.EXPT_SIM}
        # Give all datasets identical times so that they can be concatted along this dim.
        times = np.arange(
            np.datetime64('2020-01-01T04:00:00'),
            np.datetime64('2020-01-11T04:00:00'),
            np.timedelta64(1, 'h'),
            dtype='datetime64[ns]',
        )

        # See comment in PlotAllCasesGeopotSpreadError.
        # tricky_code
        expt_eRMSE = {}
        expt_dRMSE = {}
        for expt in conf.EXPT_SIM:
            case_expt_eRMSEs = []
            case_expt_dRMSEs = []
            for case in conf.CASES:
                # TODO:
                if '04' in case:
                    # There is an issue with the data for April for u-dg135.
                    # Skip for now.
                    continue
                case_expt_eRMSE = xr.load_dataarray(inputs[f'{case}_{expt}_eRMSE'])
                case_expt_dRMSE = xr.load_dataarray(inputs[f'{case}_{expt}_dRMSE'])
                case_expt_eRMSE.time.values[:] = times
                case_expt_dRMSE.time.values[:] = times
                case_expt_eRMSEs.append(case_expt_eRMSE)
                case_expt_dRMSEs.append(case_expt_dRMSE)
            expt_eRMSE[expt] = xr.concat(case_expt_eRMSEs, dim='case').mean(dim='case')
            expt_dRMSE[expt] = xr.concat(case_expt_dRMSEs, dim='case').mean(dim='case')
        plot_spread_error_ts(expt_dRMSE, expt_eRMSE, **plot_kwargs)
        plt.savefig(outputs['fig'])

        plot_summary_spread_error_ts(expt_dRMSE, expt_eRMSE, ens=plot_kwargs.get('ens', 'red'))
        plt.savefig(outputs['summary_fig'])


class CalcAutocorrImerg(Rule):
    # enabled = False
    @staticmethod
    def rule_inputs(case):
        inputs = {'imerg': RegridImergToN216.rule_outputs(case, 'cons')['output']}
        return inputs

    @staticmethod
    def rule_outputs(case):
        args, kwargs = conf.UM_TIMES[case]
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'imerg_autocorr': (
                conf.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/N216.cons.autocorr_3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    rule_matrix = {'case': conf.CASES}

    @staticmethod
    def rule_run(inputs, outputs, case):
        imerg = xr.load_dataarray(inputs['imerg'])
        nlat = len(imerg.latitude)
        nlon = len(imerg.longitude)

        dsout = imerg[0].copy().rename('imerg_autocorr').to_dataset()
        dsout['imerg_autocorr_pvalue'] = imerg[0].copy().rename('imerg_autocorr_pvalue')

        for j, k in product(range(nlat), range(nlon)):
            v = imerg[:, j, k].values
            # TODO: is this the right thing to do?
            v[np.isnan(v)] = 0
            res = scipy.stats.pearsonr(v[:-1], v[1:])
            dsout.imerg_autocorr[j, k] = res[0]
            dsout.imerg_autocorr_pvalue[j, k] = res[1]

        utils.to_netcdf_tmp_then_copy(dsout, outputs['imerg_autocorr'])


class CalcAutocorrExpt(Rule):
    # enabled = False
    @staticmethod
    def rule_inputs(expt, case):
        suite = conf.EXPT_SIM[expt]
        inputs = {
            'pflux': (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/' f'englaa_pa.merged.{case}.{suite}.m01s05i216.nc'
            )
        }
        return inputs

    @staticmethod
    def rule_outputs(expt, case):
        suite = conf.EXPT_SIM[expt]
        outputs = {'pflux_autocorr': conf.SIMDIR / f'{suite}/processed/{expt}/{case}/engla_pa.autocorr_precip.nc'}
        return outputs

    rule_matrix = {
        'expt': list(conf.EXPT_SIM.keys()),
        'case': conf.CASES,
    }

    @staticmethod
    def rule_run(inputs, outputs, expt, case):
        pflux = xr.load_dataset(inputs['pflux']).precipitation_flux.load()
        pflux.values *= 3600
        pflux.attrs['units'] = 'mm h-1'
        nens = len(pflux.realization)
        nlat = len(pflux.latitude)
        nlon = len(pflux.longitude)

        dsout = pflux[:, 0].copy().rename('pflux_autocorr').to_dataset()
        dsout['pflux_autocorr_pvalue'] = pflux[:, 0].copy().rename('pflux_autocorr_pvalue')

        for i, j, k in product(range(nens), range(nlat), range(nlon)):
            if j == k == 0:
                print(i)
            v = pflux[i, :, j, k].values
            res = scipy.stats.pearsonr(v[:-1], v[1:])
            dsout.pflux_autocorr[i, j, k] = res[0]
            dsout.pflux_autocorr_pvalue[i, j, k] = res[1]

        utils.to_netcdf_tmp_then_copy(dsout, outputs['pflux_autocorr'])


class PlotAutocorr(Rule):
    # enabled = False
    @staticmethod
    def rule_inputs(case):
        inputs = {}
        inputs['imerg'] = CalcAutocorrImerg.rule_outputs(case)['imerg_autocorr']
        for expt in conf.EXPT_SIM:
            inputs[expt] = CalcAutocorrExpt.rule_outputs(expt, case)['pflux_autocorr']
        return inputs

    @staticmethod
    def rule_outputs(case):
        return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / f'precip_autocorr.{case}.cons.png'}

    rule_matrix = {
        'case': conf.CASES,
    }

    @staticmethod
    def rule_run(inputs, outputs, case):
        imerg_ac = xr.load_dataset(inputs['imerg'])
        expts_ac = {expt: xr.load_dataset(inputs[expt]) for expt in conf.EXPT_SIM}

        fig, axes = plt.subplots(4, 4, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
        fig.set_size_inches((24, 12))
        im = axes[0, 0].pcolormesh(imerg_ac.longitude, imerg_ac.latitude, imerg_ac.imerg_autocorr, vmin=0, vmax=1)
        # plt.colorbar(im, ax=axes[0])
        axes[0, 0].coastlines()

        for ax, expt_ac in zip(axes[0, 1:], expts_ac.values()):
            im = ax.pcolormesh(
                expt_ac.longitude, expt_ac.latitude, expt_ac.pflux_autocorr.mean(dim='realization'), vmin=0, vmax=1
            )
            # plt.colorbar(im, ax=ax)
            ax.coastlines()

        plt.colorbar(im, ax=axes[0, -1])

        for ax, expt_ac in zip(axes[1, 1:], expts_ac.values()):
            im = ax.pcolormesh(
                expt_ac.longitude,
                expt_ac.latitude,
                expt_ac.pflux_autocorr.mean(dim='realization') - imerg_ac.imerg_autocorr,
                vmin=-0.4,
                vmax=0.4,
                cmap='bwr',
            )
            # plt.colorbar(im, ax=ax)
            ax.coastlines()

        plt.colorbar(im, ax=axes[1, -1])

        for ax, expt_ac in zip(axes[2, 2:], list(expts_ac.values())[1:]):
            im = ax.pcolormesh(
                expt_ac.longitude,
                expt_ac.latitude,
                expt_ac.pflux_autocorr.mean(dim='realization')
                - expts_ac['ctrl'].pflux_autocorr.mean(dim='realization'),
                vmin=-0.2,
                vmax=0.2,
                cmap='bwr',
            )
            # plt.colorbar(im, ax=ax)
            ax.coastlines()

        plt.colorbar(im, ax=axes[2, -1])

        ax = axes[3, 3]
        im = ax.pcolormesh(
            expt_ac.longitude,
            expt_ac.latitude,
            expts_ac['vanillaMCSP'].pflux_autocorr.mean(dim='realization')
            - expts_ac['stochMCSP'].pflux_autocorr.mean(dim='realization'),
            vmin=-0.2,
            vmax=0.2,
            cmap='bwr',
        )
        # plt.colorbar(im, ax=ax)
        ax.coastlines()

        plt.colorbar(im, ax=ax)

        for ax in axes[np.tril_indices(4, -1)].flatten():
            ax.axis('off')

        axes[0, 0].set_title('imerg')
        for ax, expt in zip(axes[0, 1:], expts_ac):
            ax.set_title(expt)

        for ax, expt in zip(axes[1, 1:], expts_ac):
            ax.set_title(f'{expt} - imerg')

        for ax, expt in zip(axes[2, 2:], list(expts_ac.keys())[1:]):
            ax.set_title(f'{expt} - ctrl')

        axes[3, 3].set_title('stochMCSP - vanillaMCSP')

        plt.savefig(outputs['fig'])


class CalcTCWV(Rule):
    @staticmethod
    def rule_inputs(case):
        inputs = {}
        for expt in conf.EXPT_SIM:
            suite = conf.EXPT_SIM[expt]
            inputs[f'tcwv_{expt}'] = (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/'
                f'englaa_pa.merged.{case}.{suite}.m01s30i461.1h-mean.nc'
            )
        inputs['tcwv_era5'] = RegridERA5ToN216.rule_outputs(case, 'tcwv')['output']
        return inputs

    @staticmethod
    def rule_outputs(case):
        return {'tcwv_output': conf.PATHS['figdir'] / 'N216sims' / case / f'tcwv.{case}.cons.nc'}

    rule_matrix = {'case': conf.CASES}

    @staticmethod
    def rule_run(inputs, outputs, case):
        e5ds = xr.open_dataset(inputs['tcwv_era5'])
        dsout = xr.Dataset()
        dsout['ERA5_tcwv'] = e5ds.__xarray_dataarray_variable__.mean(dim=['time'])
        for expt in conf.EXPT_SIM:
            print(expt)
            ds = xr.open_dataset(inputs[f'tcwv_{expt}'])
            tcwv = ds.m01s30i461.rename('tcwv')
            dsout[f'{expt}_tcwv'] = tcwv.mean(dim=['realization', 'time'])
        utils.to_netcdf_tmp_then_copy(dsout, outputs['tcwv_output'])


class PlotTCWV(Rule):
    rule_inputs = CalcTCWV.rule_outputs

    @staticmethod
    def rule_outputs(case):
        return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / f'tcwv.{case}.cons.png'}

    rule_matrix = {'case': conf.CASES}

    @staticmethod
    def rule_run(inputs, outputs, case):
        # e5ds = xr.open_dataset(inputs['tcwv_era5'])
        # tcwv_mean = {
        #     'ERA5': e5ds.__xarray_dataarray_variable__.mean(dim=['time']),
        # }
        # for expt in conf.EXPT_SIM:
        #     print(expt)
        #     ds = xr.open_dataset(inputs[f'tcwv_{expt}'])
        #     tcwv = ds.m01s30i461.rename('tcwv')
        #     tcwv_mean[expt] = tcwv.mean(dim=['realization', 'time'])

        ds = xr.load_dataset(inputs['tcwv_output'])
        fig, axes = plt.subplots(
            4, 4, figsize=(22, 10), subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained'
        )

        norm = mpl.colors.BoundaryNorm(np.arange(0, 90, 10), ncolors=256)
        for ax, (expt, tcwv) in zip(axes[0, :], ds.items()):
            im = ax.pcolormesh(tcwv.longitude, tcwv.latitude, tcwv, norm=norm)
            ax.coastlines()
            ax.set_title(expt)
        plt.colorbar(im, ax=axes[0], label='TCWV (mm)')

        norm2 = mpl.colors.BoundaryNorm([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16], ncolors=256)
        for ax, (expt, tcwv) in zip(axes[1, 1:], list(ds.items())[1:]):
            im = ax.pcolormesh(tcwv.longitude, tcwv.latitude, tcwv - ds['ERA5_tcwv'], norm=norm2, cmap='bwr')
            ax.coastlines()
        plt.colorbar(im, ax=axes[1], label=r'$\Delta$ TCWV (mm)')

        for ax, (expt, tcwv) in zip(axes[2, 2:], list(ds.items())[2:]):
            im = ax.pcolormesh(tcwv.longitude, tcwv.latitude, tcwv - ds['ctrl_tcwv'], norm=norm2, cmap='bwr')
            ax.coastlines()
        plt.colorbar(im, ax=axes[2], label=r'$\Delta$ TCWV (mm)')

        for ax, (expt, tcwv) in zip(axes[3, 3:], list(ds.items())[3:]):
            im = ax.pcolormesh(tcwv.longitude, tcwv.latitude, tcwv - ds['vanillaMCSP_tcwv'], norm=norm2, cmap='bwr')
            ax.coastlines()
        plt.colorbar(im, ax=axes[3], label=r'$\Delta$ TCWV (mm)')

        for ax in axes[np.tril_indices(4, -1)].flatten():
            ax.axis('off')

        plt.savefig(outputs['fig'])


class CalcMCSPCallingFreqData(Rule):
    rule_matrix = {'case': conf.CASES}

    @staticmethod
    def rule_inputs(case):
        inputs = {}
        for expt in ['vanillaMCSP', 'stochMCSP']:
            suite = conf.EXPT_SIM[expt]
            inputs[f'cf_{expt}'] = (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/'
                f'englaa_pa.merged.{case}.{suite}.m01s05i993.1h-mean.nc'
            )
            inputs[f'pflux_{expt}'] = (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/'
                f'englaa_pa.merged.{case}.{suite}.m01s05i216.1h-mean.nc'
            )
        return inputs

    @staticmethod
    def rule_outputs(case):
        return {'mcsp_calling_freq': conf.PATHS['outdir'] / 'N216sims' / case / f'MCSP_calling_freq.{case}.nc'}

    @staticmethod
    def rule_run(inputs, outputs, case):
        expt_precip = {}
        expt_cf = {}
        for expt in ['vanillaMCSP', 'stochMCSP']:
            print(expt)
            expt_precip[expt] = xr.load_dataset(inputs[f'pflux_{expt}']).precipitation_flux
            expt_precip[expt].values *= 3600
            expt_precip[expt].attrs['units'] = 'mm h-1'
            expt_cf[expt] = xr.load_dataset(inputs[f'cf_{expt}']).m01s05i993

        cfmean_vanillaMCSP = expt_cf['vanillaMCSP'].isel(realization=slice(1, 10)).mean(['realization', 'time'])
        # cfmean_vanillaMCSP.rename('cfmean_vanillaMCSP')
        cfmean_stochMCSP = expt_cf['stochMCSP'].isel(realization=slice(1, 10)).mean(['realization', 'time'])
        # cfmean_stochMCSP.rename('cfmean_stochMCSP')
        precipmean_vanillaMCSP = expt_precip['vanillaMCSP'].isel(realization=slice(1, 10)).mean(['realization', 'time'])
        # precipmean_vanillaMCSP.rename('precipmean_vanillaMCSP')
        precipmean_stochMCSP = expt_precip['stochMCSP'].isel(realization=slice(1, 10)).mean(['realization', 'time'])
        # precipmean_stochMCSP.rename('precipmean_stochMCSP')

        ds = xr.Dataset()
        ds['cfmean_vanillaMCSP'] = cfmean_vanillaMCSP
        ds['cfmean_stochMCSP'] = cfmean_stochMCSP
        ds['precipmean_vanillaMCSP'] = precipmean_vanillaMCSP
        ds['precipmean_stochMCSP'] = precipmean_stochMCSP

        utils.to_netcdf_tmp_then_copy(ds, outputs['mcsp_calling_freq'])


class PlotMCSPCallingFreq(Rule):
    rule_matrix = {'case': conf.CASES}

    rule_inputs = CalcMCSPCallingFreqData.rule_outputs

    @staticmethod
    def rule_outputs(case):
        return {'fig': conf.PATHS['figdir'] / 'N216sims' / case / f'MCSP_calling_freq.{case}.png'}

    @staticmethod
    def rule_run(inputs, outputs, case):
        ds = xr.load_dataset(inputs['mcsp_calling_freq'])

        cfmean_vanillaMCSP = ds['cfmean_vanillaMCSP']
        cfmean_stochMCSP = ds['cfmean_stochMCSP']
        cfmeans = {
            'vanillaMCSP': cfmean_vanillaMCSP,
            'stochMCSP': cfmean_stochMCSP,
        }

        precipmean_vanillaMCSP = ds['precipmean_vanillaMCSP']
        precipmean_stochMCSP = ds['precipmean_stochMCSP']
        precipmeans = {
            'vanillaMCSP': precipmean_vanillaMCSP,
            'stochMCSP': precipmean_stochMCSP,
        }

        fig, axes = plt.subplots(
            2, 3, figsize=(25.5, 8), subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained'
        )

        for ax, expt in zip(axes[:, 0], ['vanillaMCSP', 'stochMCSP']):
            precipmean = precipmeans[expt]
            ax.set_title(f'precip. {expt}')
            ax.coastlines()
            # bounds = np.arange(0, np.round(cfmax + 0.1, 1), 0.05)
            bounds = [0, 0.125, 0.25, 0.5, 1, 2, 3, 4]
            cmap = mpl.cm.viridis
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
            im = ax.pcolormesh(precipmean.longitude, precipmean.latitude, precipmean, norm=norm, cmap=cmap)
            plt.colorbar(im, ax=ax, label='precip. (mm h$^{-1}$)')

        for ax, expt in zip(axes[:, 1], ['vanillaMCSP', 'stochMCSP']):
            cfmean = cfmeans[expt]
            precipmean = precipmeans[expt]
            pcc = scipy.stats.pearsonr(cfmean.values.flatten(), precipmean.values.flatten())[0]
            ax.set_title(f'MCSP calling freq. {expt} (PCC with precip.: {pcc:.3f})')
            ax.coastlines()
            # bounds = np.arange(0, np.round(cfmax + 0.1, 1), 0.05)
            bounds = [0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
            cmap = mpl.cm.viridis
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
            im = ax.pcolormesh(cfmean.longitude, cfmean.latitude, cfmean, norm=norm, cmap=cmap)
            plt.colorbar(im, ax=ax, label='calling freq. (frac)')

        for ax, method in zip(axes[:, 2], ['diff', 'frac']):
            ax.coastlines()
            if method == 'diff':
                cfdata = cfmean_vanillaMCSP - cfmean_stochMCSP
                ax.set_title(f'MCSP calling freq. vanillaMCSP - stochMCSP (mean={np.nanmean(cfdata):.3f})')
                absmax = np.round(np.abs(cfdata.values).max(), 1)
                bounds = np.arange(-absmax, absmax, 0.1)
                cmap = mpl.cm.bwr
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
                im = ax.pcolormesh(cfdata.longitude, cfdata.latitude, cfdata, norm=norm, cmap=cmap)
                plt.colorbar(im, ax=ax, label='calling freq. (frac)')
            elif method == 'frac':
                cfdata = np.ma.masked_invalid(cfmean_vanillaMCSP / cfmean_stochMCSP)
                cfdata_mean = cfdata.mean()
                ax.set_title(f'MCSP calling freq. vanillaMCSP / stochMCSP (mean={cfdata_mean:.3f})')
                # cf99 = np.nanpercentile(cfdata.values, 99)
                # print(np.max(cfdata))
                bounds = [0, 0.1, 0.2, 0.5, 0.9, 1.1, 2, 5, 10, 20]
                cmap = mpl.cm.bwr
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
                ax.set_facecolor('grey')
                im = ax.pcolormesh(
                    cfmean_vanillaMCSP.longitude, cfmean_vanillaMCSP.latitude, cfdata, norm=norm, cmap=cmap
                )
                plt.colorbar(
                    im, ax=ax, label='calling freq. (frac)', extend='max', ticks=[0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
                )

        plt.savefig(outputs['fig'])


class FirstLookPlotERA5_500hPa_geopotential(Rule):
    rule_matrix = {
        'case': conf.CASES,
        'time_offset': [0, 1, 6, 24, 48, 120, 239],
        # 'time_offset': [0, 1, 6, 24, 48, 120],
    }

    @staticmethod
    def rule_inputs(case, time_offset):
        month = case[4:6]
        inputs = {
            'era5geopot': (
                conf.PATHS['datadir']
                / f'ecmwf-era5/misc/2020/{month}'
                / f'ecmwf-era5_oper_an_pl.2024.{month}.1-11.geopotential_500hPa.nc'
            ),
        }
        for expt, sim in conf.EXPT_SIM.items():
            inputs[f'{expt}geopot'] = (
                conf.SIMDIR / f'{sim}/share/cycle/{case}/engl/um/englaa_pb.merged.{case}.{sim}.m01s16i202.500hPa.nc'
            )
        return inputs

    @staticmethod
    def rule_outputs(case, time_offset):
        times_args = conf.UM_TIMES[case]
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')

        outputs = {
            'era5fig': conf.PATHS['figdir'] / 'N216sims' / case / 'geopot' / f'era5.geopot.{d0}.{time_offset}h.png',
        }
        for expt, sim in conf.EXPT_SIM.items():
            for i in range(10):
                outputs[f'{expt}fig_{i}'] = (
                    conf.SIMDIR
                    / conf.PATHS['figdir']
                    / 'N216sims'
                    / case
                    / 'geopot'
                    / f'{expt}.geopot.{d0}.em{i:02d}.{time_offset}h.png'
                )
        return outputs

    @staticmethod
    def rule_run(inputs, outputs, case, time_offset):
        g = 9.80665
        levels = np.arange(460, 624, 4)
        lw = [1 if l % 20 != 0 else 2 for l in levels]

        def fmt(x):
            s = f"{x:.0f}"
            return s

        print('- era5geopot')
        era5geopot = xr.open_dataarray(inputs['era5geopot']).isel(valid_time=4 + time_offset).load() / g

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(30, 15), layout='constrained')
        # /10 converts to dam.
        cs = ax.contour(
            era5geopot.longitude,
            era5geopot.latitude,
            era5geopot.sel(pressure_level=500) / 10,
            levels=levels,
            linewidths=lw,
            colors='k',
        )
        ax.coastlines(color='grey')

        ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
        plt.savefig(outputs['era5fig'])

        for expt, sim in conf.EXPT_SIM.items():
            print(f'- {expt}geopot')
            for i in range(10):
                simgeopot = xr.open_dataset(inputs[f'{expt}geopot']).geopotential_height
                simgeopot = simgeopot.sel(realization=i).isel(time=0 + time_offset)

                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(30, 15), layout='constrained'
                )
                try:
                    # Occasionally there's an error deep in this call.
                    # /10 converts to dam.
                    cs = ax.contour(
                        simgeopot.longitude,
                        simgeopot.latitude,
                        simgeopot / 10,
                        levels=levels,
                        linewidths=lw,
                        colors='k',
                    )
                    ax.coastlines(color='grey')

                    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
                    plt.savefig(outputs[f'{expt}fig_{i}'])
                except:
                    outputs[f'{expt}fig_{i}'].touch()
                plt.close('all')


class Calc_geopot_dRMSE(Rule):
    rule_matrix = {
        'expt': list(conf.EXPT_SIM.keys()),
        'case': conf.CASES,
        'domain': ['global', 'tropics'],
    }

    @staticmethod
    def rule_inputs(expt, case, domain):
        suite = conf.EXPT_SIM[expt]
        inputs = {
            'geopot': (
                conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/englaa_pb.merged.{case}.{suite}.m01s16i202.500hPa.nc'
            )
        }
        return inputs

    @staticmethod
    def rule_outputs(expt, case, domain):
        suite = conf.EXPT_SIM[expt]
        outputs = {'dRMSE': conf.SIMDIR / f'{suite}/processed/{expt}/{case}/engla_pb.geopot.{domain}.dRMSE.nc'}
        return outputs

    @staticmethod
    def rule_run(inputs, outputs, expt, case, domain):
        if domain == 'tropics':
            simgeopot = xr.open_dataset(inputs['geopot']).sel(latitude=slice(-30, 30)).geopotential_height.load()
        elif domain == 'global':
            simgeopot = xr.open_dataset(inputs['geopot']).geopotential_height.load()
        print(simgeopot)
        ntime = len(simgeopot.time)
        dRMSE_data = np.full((nens, nens, ntime), np.nan)

        for ens_idx1 in range(nens):
            for ens_idx2 in range(ens_idx1 + 1, nens):
                print(expt, ens_idx1, ens_idx2)
                for time_idx in range(ntime):
                    dRMSE_data[ens_idx1, ens_idx2, time_idx] = rmse(
                        simgeopot[ens_idx1, time_idx], simgeopot[ens_idx2, time_idx]
                    )
        dRMSE = xr.DataArray(
            dRMSE_data,
            coords={
                'realization1': simgeopot['realization'].values,
                'realization2': simgeopot['realization'].values,
                'time': simgeopot['time'],
            },
        )
        utils.to_netcdf_tmp_then_copy(dRMSE, outputs['dRMSE'])


class Calc_geopot_eRMSE(Rule):
    rule_matrix = {
        'expt': list(conf.EXPT_SIM.keys()),
        'case': conf.CASES,
        'domain': ['global', 'tropics'],
    }

    @staticmethod
    def rule_inputs(expt, case, domain):
        suite = conf.EXPT_SIM[expt]
        month = case[4:6]
        if domain == 'global':
            inputs = {
                'era5geopot': RegridERA5ToN216.rule_outputs(case, 'z')['output'],
                'simgeopot': (
                    conf.SIMDIR
                    / f'{suite}/share/cycle/{case}/engl/um/englaa_pb.merged.{case}.{suite}.m01s16i202.500hPa.nc'
                ),
            }
        elif domain == 'tropics':
            inputs = {
                'era5geopot': RegridERA5ToN216.rule_outputs(case, 'z')['output'],
                'simgeopot': (
                    conf.SIMDIR
                    / f'{suite}/share/cycle/{case}/engl/um/englaa_pb.merged.{case}.{suite}.m01s16i202.500hPa.nc'
                ),
            }
        return inputs

    @staticmethod
    def rule_outputs(expt, case, domain):
        suite = conf.EXPT_SIM[expt]
        outputs = {'eRMSE': conf.SIMDIR / f'{suite}/processed/{expt}/{case}/engla_pb.geopot.{domain}.eRMSE.nc'}
        return outputs

    @staticmethod
    def rule_run(inputs, outputs, expt, case, domain):
        g = 9.80665
        print(expt, case, domain)
        # Using regridded ERA5 now.
        # TODO: Need to regrid ERA5 first, but amazingly RMSE seems to work even tho they're
        # on different grids??!!??
        # Why is the latitude the other way round on this???
        if domain == 'tropics':
            simgeopot = xr.open_dataset(inputs['simgeopot']).sel(latitude=slice(-30, 30)).geopotential_height.load()
            # Rem. ERA5 starts at 00:00, sims at 04:00.
            era5geopot = (
                xr.load_dataarray(inputs['era5geopot']).sel(latitude=slice(-30, 30)).isel(valid_time=slice(4, None)) / g
            )
        elif domain == 'global':
            simgeopot = xr.open_dataset(inputs['simgeopot']).geopotential_height.load()
            # # Rem. ERA5 starts at 00:00, sims at 04:00.
            era5geopot = xr.load_dataarray(inputs['era5geopot']).isel(valid_time=slice(4, None)) / g
        print(simgeopot)
        print(era5geopot)
        ntime = len(simgeopot.time)
        eRMSE_data = np.full((nens, ntime), np.nan)

        for ens_idx in range(nens):
            # print(expt, ens_idx)
            for time_idx in range(ntime):
                eRMSE_data[ens_idx, time_idx] = rmse(era5geopot[time_idx], simgeopot[ens_idx, time_idx])

        eRMSE = xr.DataArray(
            eRMSE_data,
            coords={
                'realization': simgeopot['realization'],
                'time': simgeopot['time'],
            },
        )
        utils.to_netcdf_tmp_then_copy(eRMSE, outputs['eRMSE'])


def plot_geopot_spread_error_ts(
    expt_dRMSE, expt_eRMSE, smooth=False, xlim='full', show_error_minus_spread=False, ens='full'
):
    ntime = len(expt_eRMSE['ctrl'].time)

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(8, 6)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # if smooth:
    #     ax.set_title(rf'$\sigma=${sigma} ({smooth} h smoothing)')
    # else:
    #     ax.set_title(rf'$\sigma=${sigma}')
    for i, expt in enumerate(conf.EXPT_SIM):
        # dRMSE_ts = np.nanmean(expt_dRMSE[expt], axis=(0, 1))
        # eRMSE_ts = np.nanmean(expt_eRMSE[expt], axis=0)
        dRMSE = expt_dRMSE[expt]
        if ens == 'red':
            dRMSE = dRMSE.sel(realization1=slice(1, 10), realization2=slice(1, 10))
            expt_eRMSE[expt] = expt_eRMSE[expt].sel(realization=slice(1, 10))

        dRMSE_ts = dRMSE.mean(dim=['realization1', 'realization2']).values
        eRMSE_ts = expt_eRMSE[expt].mean(dim=['realization']).values
        if smooth:
            dRMSE_ts = np.convolve(dRMSE_ts, np.ones((smooth,)) / smooth, mode='same')
            eRMSE_ts = np.convolve(eRMSE_ts, np.ones((smooth,)) / smooth, mode='same')
        c = cs[i]
        ax.plot(eRMSE_ts, color=c, label=f'{expt} error')
        ax.plot(dRMSE_ts, ls='--', color=c, label=f'{expt} spread')
        # if show_error_minus_spread:
        #     ax.plot(eRMSE_ts - dRMSE_ts, ls=':', color=c, label=f'{expt} error - spread')
        #     ax.axhline(y=0, ls='-', lw=0.5, color='k')

    times_half_days_hours = np.arange(0, ntime + 1, 12)
    times_days = [int(v) for v in times_half_days_hours / 24]
    times_days[1::2] = [''] * len(times_days[1::2])
    ax.set_xticks(times_half_days_hours, times_days)
    ax.set_xlim((0, ntime))

    ax.set_ylabel('RMSE (dam)')
    if xlim != 'full':
        ax.set_xlim(xlim)
    if smooth and xlim == 'full':
        ax.set_xlim((smooth, 240 - smooth))
    ax.relim()
    # if not show_error_minus_spread:
    #     ax.set_ylim((0, None))

    ax.legend(ncol=len(conf.EXPT_SIM))
    ax.set_xlabel('time (day)')


class PlotGeopotSpreadError(Rule):
    rule_matrix = {
        'plot_kwargs': [
            dict(smooth=False, show_error_minus_spread=True),
            dict(smooth=24),
            dict(xlim=(0, 20)),
            dict(xlim=(0, 48)),
            dict(smooth=False, show_error_minus_spread=True, ens='full'),
            dict(smooth=False, show_error_minus_spread=True, ens='red'),
        ],
        'case': conf.CASES,
        'domain': ['global', 'tropics'],
    }

    @staticmethod
    def rule_inputs(plot_kwargs, case, domain):
        inputs = {
            f'{expt}_eRMSE': Calc_geopot_eRMSE.rule_outputs(expt, case, domain)['eRMSE'] for expt in conf.EXPT_SIM
        }
        inputs.update(
            {f'{expt}_dRMSE': Calc_geopot_dRMSE.rule_outputs(expt, case, domain)['dRMSE'] for expt in conf.EXPT_SIM}
        )
        return inputs

    @staticmethod
    def rule_outputs(plot_kwargs, case, domain):
        kwstr = '-'.join(f'{k}={v}' for k, v in plot_kwargs.items())
        kwstr = kwstr.replace(' ', '')
        return {
            'fig': (
                conf.PATHS['figdir'] / 'N216sims' / case / 'geopot' / f'geopot_spread_error.{case}.{domain}.{kwstr}.png'
            ),
        }

    @staticmethod
    def rule_run(inputs, outputs, plot_kwargs, case, domain):
        print(case, plot_kwargs)
        expt_eRMSE = {expt: xr.load_dataarray(inputs[f'{expt}_eRMSE']) for expt in conf.EXPT_SIM}
        expt_dRMSE = {expt: xr.load_dataarray(inputs[f'{expt}_dRMSE']) for expt in conf.EXPT_SIM}
        plot_geopot_spread_error_ts(expt_dRMSE, expt_eRMSE, **plot_kwargs)
        plt.savefig(outputs['fig'])


class PlotAllCasesGeopotSpreadError(Rule):
    rule_matrix = {
        'plot_kwargs': [
            dict(smooth=False, show_error_minus_spread=True),
            dict(smooth=24),
            dict(xlim=(0, 20)),
            dict(xlim=(0, 48)),
            dict(smooth=False, show_error_minus_spread=True, ens='full'),
            dict(smooth=False, show_error_minus_spread=True, ens='red'),
        ],
        'domain': ['global', 'tropics'],
    }

    @staticmethod
    def rule_inputs(plot_kwargs, domain):
        inputs = {
            f'{case}_{expt}_eRMSE': Calc_geopot_eRMSE.rule_outputs(expt, case, domain)['eRMSE']
            for expt in conf.EXPT_SIM
            for case in conf.CASES
        }
        inputs.update(
            {
                f'{case}_{expt}_dRMSE': Calc_geopot_dRMSE.rule_outputs(expt, case, domain)['dRMSE']
                for expt in conf.EXPT_SIM
                for case in conf.CASES
            }
        )
        return inputs

    @staticmethod
    def rule_outputs(plot_kwargs, domain):
        kwstr = '-'.join(f'{k}={v}' for k, v in plot_kwargs.items())
        kwstr = kwstr.replace(' ', '')
        return {
            'fig': (
                conf.PATHS['figdir']
                / 'N216sims'
                / 'all_cases'
                / 'geopot'
                / f'geopot_spread_error.all_cases.{domain}.{kwstr}.png'
            ),
        }

    @staticmethod
    def rule_run(inputs, outputs, plot_kwargs, domain):
        print(plot_kwargs, domain)

        # Give all datasets identical times so that they can be concatted along this dim.
        times = np.arange(
            np.datetime64('2020-01-01T04:00:00'),
            np.datetime64('2020-01-11T04:00:00'),
            np.timedelta64(1, 'h'),
            dtype='datetime64[ns]',
        )

        # tricky_code
        # Getting this right took a fair bit of iteration.
        # Idea: load in all cases for each expt, and reset their times (to those of the first).
        # Then concat along a new dim, case. If you don't reset their times, then the concat will
        # not do what I want, because each time (different for each file) will be offset from all others.
        # i.e. datasets will have 2880 values along time, instead of 288.
        expt_eRMSE = {}
        expt_dRMSE = {}
        for expt in conf.EXPT_SIM:
            case_expt_eRMSEs = []
            case_expt_dRMSEs = []
            for case in conf.CASES:
                # TODO:
                if '04' in case:
                    # There is an issue with the data for April for u-dg135.
                    # Skip for now.
                    continue
                case_expt_eRMSE = xr.load_dataarray(inputs[f'{case}_{expt}_eRMSE'])
                case_expt_dRMSE = xr.load_dataarray(inputs[f'{case}_{expt}_dRMSE'])
                case_expt_eRMSE.time.values[:] = times
                case_expt_dRMSE.time.values[:] = times
                case_expt_eRMSEs.append(case_expt_eRMSE)
                case_expt_dRMSEs.append(case_expt_dRMSE)
            expt_eRMSE[expt] = xr.concat(case_expt_eRMSEs, dim='case').mean(dim='case')
            expt_dRMSE[expt] = xr.concat(case_expt_dRMSEs, dim='case').mean(dim='case')

        # import IPython
        # IPython.embed(colors='neutral')
        plot_geopot_spread_error_ts(expt_dRMSE, expt_eRMSE, **plot_kwargs)
        plt.savefig(outputs['fig'])
