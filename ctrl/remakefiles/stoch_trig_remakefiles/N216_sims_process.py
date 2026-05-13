import pandas as pd
from itertools import product

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
import scipy.stats
import xarray as xr
import xesmf as xe

if __remake__ == '__old__':
    from remake import Remake, TaskRule
elif __remake__ == '__new__':
    from remake2 import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'
N_ENS_MEM = 10

EXPT_SIM = {
    'Control': 'u-di727',
    'vanillaMCSP': 'u-di728',
    'STOCH-PRIME-MCSP': 'u-dg135',
}


IMERG_FINAL_30MIN_DIR = DATADIR / 'GPM_IMERG_final/30min'
# These times exactly match the UM sims.
# UM_TIMES = pd.date_range('2020-07-01 04:00', '2020-07-11 03:00', freq='H')
UM_TIMES = (('2020-07-01 04:00', '2020-07-11 03:00'), {'freq': 'H'}) # args, kwargs for date_range.

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))


class N216ExtractCombinePrecip(TaskRule):
    """Extract and combine precipitation at all times and for all EMs.
    """
    @staticmethod
    def rule_inputs(expt):
        suite = EXPT_SIM[expt]
        inputs = {
            f'pa_em{em_idx}_{h:03d}': SIMDIR / f'{suite}/share/cycle/20200701T0000Z/engl/um/em{em_idx}/englaa_pa{h:03d}.iris.nc'
            for em_idx in range(N_ENS_MEM)
            for h in range(0, 217, 24)
        }
        return inputs

    @staticmethod
    def rule_outputs(expt):
        suite = EXPT_SIM[expt]
        outputs = {
            f'pflux': SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'
        }
        return outputs


    var_matrix = {
        'expt': list(EXPT_SIM.keys())
    }

    def rule_run(self):
        def load_em_pflux(inputs, ens_idx):
            keep_coords = ['time', 'latitude', 'longitude']
            time_pfluxes = []

            for h in range(0, 217, 24):
                self.logger.debug(f'  Opening time {h}')

                em_path = inputs[f'pa_em{ens_idx}_{h:03d}']
                dsa = xr.open_dataset(em_path)
                # What's the difference between the two fluxes?
                # dsa.precipitation_flux has no time mean (i.e. it's instantaneous)
                # dsa.precipitation_flux_0 has 1-hr time mean.
                # I think it's better to use instantaneous to e.g. compare with IMERG.
                pflux = dsa.precipitation_flux
                # Why does time 0 have different names for all variables?
                if h == 0:
                    pflux = pflux.rename(time_0='time')
                # Drop all variables that are not needed. This means concat will work.
                # (This drops all other coords with _0 suffix.)
                coord_names = [c.name for c in list(pflux.coords.values())]
                drop_coords = sorted(set(coord_names) - set(keep_coords))
                pflux = pflux.drop_vars(drop_coords)

                time_pfluxes.append(pflux)

            pflux = xr.concat(time_pfluxes, dim='time')
            return pflux.load()

        em_pfluxes = []
        for ens_idx in range(N_ENS_MEM):
            self.logger.info(f'Loading EM {ens_idx}')
            em_pfluxes.append(load_em_pflux(self.inputs, ens_idx))
        pflux = xr.concat(em_pfluxes, dim=pd.Index(range(N_ENS_MEM), name='ens_mem'))

        pflux.attrs['UM simulation'] = EXPT_SIM[self.expt]
        pflux.attrs['MCS:PRIME expt'] = self.expt

        cu.to_netcdf_tmp_then_copy(pflux, self.outputs['pflux'])


class RegridImergToN216(TaskRule):
    """Regrid IMERG to the same grid as N216 simulations.
    """
    @staticmethod
    def rule_inputs(times_args):
        # Just load IMERG on the hour "S??0000".
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        def h_to_m(h):
            return h * 60
        inputs = {
            f'imerg_{t}': (
                IMERG_FINAL_30MIN_DIR /
                f'{t.year}/{t.month:02d}/{t.day:02d}/' /
                f'3B-HHR.MS.MRG.3IMERG.{t.year}{t.month:02d}{t.day:02d}-S{t.hour:02d}0000-E{t.hour:02d}2959.{h_to_m(t.hour):04d}.V07B.HDF5.nc4'
            )
            for t in times
        }
        inputs['pflux'] = N216ExtractCombinePrecip.rule_outputs('STOCH-PRIME-MCSP')['pflux']

        return inputs

    @staticmethod
    def rule_outputs(times_args):
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'output': (
                cu.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    var_matrix = {
        'times_args': [UM_TIMES],
    }

    def rule_run(self):
        pflux = xr.open_dataarray(self.inputs['pflux'])
        imerg = xr.open_mfdataset([v for k, v in self.inputs.items() if k.startswith('imerg')])
        print(pflux)
        print(imerg)
        # TODO: conservative method.
        regridder = xe.Regridder(imerg.precipitation, pflux, method='bilinear')
        imerg_N216 = regridder(imerg.precipitation)
        cu.to_netcdf_tmp_then_copy(imerg_N216, self.outputs['output'])


class PlotTotalPrecip(TaskRule):
    @staticmethod
    def rule_inputs():
        inputs = {}
        for expt in EXPT_SIM:
            suite = EXPT_SIM[expt]
            inputs[f'pflux_{expt}'] = SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'
        inputs['imerg'] = RegridImergToN216.rule_outputs(UM_TIMES)['output']
        return inputs


    @staticmethod
    def rule_outputs():
        return {'fig': cu.PATHS['figdir'] / 'N216sims' / 'total_precip.png'}

    def rule_run(self):
        expt_pflux = {}
        for expt in EXPT_SIM:
            pflux = xr.load_dataarray(self.inputs[f'pflux_{expt}'])
            pflux.values *= 3600
            pflux.attrs['units'] = 'mm h-1'
            expt_pflux[expt] = pflux
        imerg = xr.load_dataarray(self.inputs['imerg'])
        imerg_ts = imerg.mean(dim=['latitude', 'longitude'])

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(range(len(imerg.time)), imerg_ts, c='k', label='IMERG')
        for expt in EXPT_SIM:
            pflux_ts = expt_pflux[expt].mean(dim=['ens_mem', 'latitude', 'longitude'])
            l, = ax.plot(range(len(pflux_ts.time)), pflux_ts, label=expt)
            ax2.plot(range(len(pflux_ts.time)), pflux_ts.values / imerg_ts.values * 100, c=l.get_color(), ls='--', label=expt)
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

        plt.savefig(self.outputs['fig'])


nens = 10
aspect = 1.43  # mean aspect ratio of cells over the tropics.
sel_tropics = {'latitude': slice(-30, 30)}
sigmas = [0, 1, 2, 4]


class GuassianFilterN216Imerg(TaskRule):
    @staticmethod
    def rule_inputs():
        inputs = {'imerg': RegridImergToN216.rule_outputs(UM_TIMES)['output']}
        return inputs

    @staticmethod
    def rule_outputs():
        args, kwargs = UM_TIMES
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'imerg_filtered': (
                cu.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/guassian_filtered_3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    def rule_run(self):
        imerg = xr.load_dataarray(self.inputs['imerg'])
        imerg_tropics = imerg.sel(**sel_tropics)
        imerg_filtered = []

        for sigma in sigmas:
            print(sigma)
            imerg_filtered_values = ndimage.gaussian_filter(imerg.sel(**sel_tropics), (0, aspect * sigma, sigma))
            da_imerg_filtered = imerg_tropics.copy()
            da_imerg_filtered.values = imerg_filtered_values
            imerg_filtered.append(da_imerg_filtered)
        imerg_filtered = xr.concat(imerg_filtered, dim=pd.Index(sigmas, name='sigma'))
        cu.to_netcdf_tmp_then_copy(imerg_filtered, self.outputs['imerg_filtered'])


class GuassianFilterExpt(TaskRule):
    @staticmethod
    def rule_inputs(expt):
        suite = EXPT_SIM[expt]
        inputs = {'pflux': SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'}
        return inputs

    @staticmethod
    def rule_outputs(expt):
        suite = EXPT_SIM[expt]
        outputs = {'pflux_filtered': SIMDIR / f'{suite}/processed/{expt}/engla_pa.filtered_precip.nc'}
        return outputs

    var_matrix = {
        'expt': list(EXPT_SIM.keys())
    }

    def rule_run(self):
        pflux = xr.load_dataarray(self.inputs['pflux'])
        pflux.values *= 3600
        pflux.attrs['units'] = 'mm h-1'
        pflux_tropics = pflux.sel(**sel_tropics)

        pflux_filtered = []
        for sigma in sigmas:
            da_pflux_filtered = pflux_tropics.copy()

            print(sigma)
            for i in range(nens):
                if sigma != 0:
                    da_pflux_filtered.values[i] = ndimage.gaussian_filter(da_pflux_filtered.values[i], (0, aspect * sigma, sigma))
            pflux_filtered.append(da_pflux_filtered)
        pflux_filtered = xr.concat(pflux_filtered, dim=pd.Index(sigmas, name='sigma'))
        cu.to_netcdf_tmp_then_copy(pflux_filtered, self.outputs['pflux_filtered'])


def rmse(a1, a2):
    return np.sqrt(np.mean((a1 - a2)**2))


class Calc_eRMSE(TaskRule):
    @staticmethod
    def rule_inputs(expt):
        inputs = {
            'pflux_filtered': GuassianFilterExpt.rule_outputs(expt)['pflux_filtered'],
            'imerg_filtered': GuassianFilterN216Imerg.rule_outputs()['imerg_filtered']
        }
        return inputs


    @staticmethod
    def rule_outputs(expt):
        suite = EXPT_SIM[expt]
        outputs = {'eRMSE': SIMDIR / f'{suite}/processed/{expt}/engla_pa.filtered_precip.eRMSE.nc'}
        return outputs

    var_matrix = {
        'expt': list(EXPT_SIM.keys())
    }

    def rule_run(self):
        pflux_filtered = xr.load_dataarray(self.inputs['pflux_filtered'])
        imerg_filtered = xr.load_dataarray(self.inputs['imerg_filtered'])
        ntime = len(pflux_filtered.time)
        eRMSE_data = np.full((len(sigmas), nens, ntime), np.nan)

        for ens_idx in range(nens):
            for sigma_idx in range(len(sigmas)):
                print(self.expt, ens_idx, sigma_idx)
                for time_idx in range(ntime):
                    eRMSE_data[sigma_idx, ens_idx, time_idx] = rmse(imerg_filtered[sigma_idx, time_idx], pflux_filtered[sigma_idx, ens_idx, time_idx])

        eRMSE = xr.DataArray(
            eRMSE_data,
            coords={
                'sigma': pflux_filtered['sigma'],
                'ens_mem': pflux_filtered['ens_mem'],
                'time': pflux_filtered['time'],
            }
        )
        cu.to_netcdf_tmp_then_copy(eRMSE, self.outputs['eRMSE'])


class Calc_dRMSE(TaskRule):
    @staticmethod
    def rule_inputs(expt):
        inputs = {
            'pflux_filtered': GuassianFilterExpt.rule_outputs(expt)['pflux_filtered'],
        }
        return inputs


    @staticmethod
    def rule_outputs(expt):
        suite = EXPT_SIM[expt]
        outputs = {'dRMSE': SIMDIR / f'{suite}/processed/{expt}/engla_pa.filtered_precip.dRMSE.nc'}
        return outputs

    var_matrix = {
        'expt': list(EXPT_SIM.keys())
    }

    def rule_run(self):
        pflux_filtered = xr.load_dataarray(self.inputs['pflux_filtered'])
        ntime = len(pflux_filtered.time)
        dRMSE_data = np.full((len(sigmas), nens, nens, ntime), np.nan)

        for ens_idx1 in range(nens):
            for ens_idx2 in range(ens_idx1 + 1, nens):
                print(self.expt, ens_idx1, ens_idx2)
                for sigma_idx in range(len(sigmas)):
                    for time_idx in range(ntime):
                        dRMSE_data[sigma_idx, ens_idx1, ens_idx2, time_idx] = rmse(
                            pflux_filtered[sigma_idx, ens_idx1, time_idx],
                            pflux_filtered[sigma_idx, ens_idx2, time_idx]
                        )
        dRMSE = xr.DataArray(
            dRMSE_data,
            coords={
                'sigma': pflux_filtered['sigma'],
                'ens_mem1': pflux_filtered['ens_mem'].values,
                'ens_mem2': pflux_filtered['ens_mem'].values,
                'time': pflux_filtered['time'],
            }
        )
        cu.to_netcdf_tmp_then_copy(dRMSE, self.outputs['dRMSE'])


plot_sigmas = [0, 2, 4]
def plot_spread_skill_ts(expt_dRMSE, expt_eRMSE, smooth=False, xlim='full', show_skill_minus_spread=False):
    ntime = len(expt_eRMSE['Control'].time)

    fig, axes = plt.subplots(1, len(plot_sigmas), sharex=True, layout='constrained')
    fig.set_size_inches(20, 6)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, sigma in zip(axes, plot_sigmas):
        if smooth:
            ax.set_title(f'$\sigma=${sigma} ({smooth} h smoothing)')
        else:
            ax.set_title(f'$\sigma=${sigma}')
        for i, expt in enumerate(EXPT_SIM):
            # dRMSE_ts = np.nanmean(expt_dRMSE[expt], axis=(0, 1))
            # eRMSE_ts = np.nanmean(expt_eRMSE[expt], axis=0)
            dRMSE_ts = expt_dRMSE[expt].mean(dim=['ens_mem1', 'ens_mem2']).sel(sigma=sigma).values
            eRMSE_ts = expt_eRMSE[expt].mean(dim=['ens_mem']).sel(sigma=sigma).values
            if smooth:
                dRMSE_ts = np.convolve(dRMSE_ts, np.ones((smooth, )) / smooth, mode='same')
                eRMSE_ts = np.convolve(eRMSE_ts, np.ones((smooth, )) / smooth, mode='same')
            c = cs[i]
            ax.plot(eRMSE_ts, color=c, label=f'{expt} skill')
            ax.plot(dRMSE_ts, ls='--', color=c, label=f'{expt} spread')
            if show_skill_minus_spread:
                ax.plot(eRMSE_ts - dRMSE_ts, ls=':', color=c, label=f'{expt} skill - spread')
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
        if not show_skill_minus_spread:
            ax.set_ylim((0, None))
    if len(axes) % 2 == 1:
        midax = axes[len(axes) // 2]
        midax.legend(ncol=len(EXPT_SIM))
        midax.set_xlabel('time (day)')
        ylim = midax.get_ylim()
        midax.set_ylim((ylim[0], ylim[1] * 1.2))
    else:
        axes[-1].legend(ncol=len(EXPT_SIM))
        for ax in axes.flatten():
            ax.set_xlabel('time (day)')

class PlotSpreadSkill(TaskRule):
    @staticmethod
    def rule_inputs(kwargs):
        inputs = {
            f'{expt}_eRMSE': Calc_eRMSE.rule_outputs(expt)['eRMSE']
            for expt in EXPT_SIM
        }
        inputs.update({
            f'{expt}_dRMSE': Calc_dRMSE.rule_outputs(expt)['dRMSE']
            for expt in EXPT_SIM
        })
        return inputs

    @staticmethod
    def rule_outputs(kwargs):
        kwstr = '-'.join(
            f'{k}={v}'
            for k, v in kwargs.items()
        )
        kwstr = kwstr.replace(' ', '')
        return {'fig': cu.PATHS['figdir'] / 'N216sims' / f'spread_skill.{kwstr}.png'}

    var_matrix = {
        'kwargs': [
            dict(smooth=False, show_skill_minus_spread=True),
            dict(smooth=24),
            dict(xlim=(0, 20)),
            dict(xlim=(0, 48)),
        ]
    }


    def rule_run(self):
        expt_eRMSE = {expt: xr.load_dataarray(self.inputs[f'{expt}_eRMSE']) for expt in EXPT_SIM}
        expt_dRMSE = {expt: xr.load_dataarray(self.inputs[f'{expt}_dRMSE']) for expt in EXPT_SIM}
        plot_spread_skill_ts(expt_dRMSE, expt_eRMSE, **self.kwargs)
        plt.savefig(self.outputs['fig'])


class CalcAutocorrImerg(TaskRule):
    @staticmethod
    def rule_inputs():
        inputs = {'imerg': RegridImergToN216.rule_outputs(UM_TIMES)['output']}
        return inputs

    @staticmethod
    def rule_outputs():
        args, kwargs = UM_TIMES
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'imerg_autocorr': (
                cu.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/autocorr_3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    def rule_run(self):
        imerg = xr.load_dataarray(self.inputs['imerg'])
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

        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['imerg_autocorr'])


class CalcAutocorrExpt(TaskRule):
    @staticmethod
    def rule_inputs(expt):
        suite = EXPT_SIM[expt]
        inputs = {'pflux': SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'}
        return inputs

    @staticmethod
    def rule_outputs(expt):
        suite = EXPT_SIM[expt]
        outputs = {'pflux_autocorr': SIMDIR / f'{suite}/processed/{expt}/engla_pa.autocorr_precip.nc'}
        return outputs

    var_matrix = {
        'expt': list(EXPT_SIM.keys())
    }

    def rule_run(self):
        pflux = xr.load_dataarray(self.inputs['pflux'])
        pflux.values *= 3600
        pflux.attrs['units'] = 'mm h-1'
        nens = len(pflux.ens_mem)
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

        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['pflux_autocorr'])


class PlotAutocorr(TaskRule):
    @staticmethod
    def rule_inputs():
        inputs = {}
        inputs['imerg'] = CalcAutocorrImerg.rule_outputs()['imerg_autocorr']
        for expt in EXPT_SIM:
            inputs[expt] = CalcAutocorrExpt.rule_outputs(expt)['pflux_autocorr']
        return inputs

    @staticmethod
    def rule_outputs():
        return {'fig': cu.PATHS['figdir'] / 'N216sims' / 'precip_autocorr.png'}

    def rule_run(self):
        imerg_ac = xr.load_dataset(self.inputs['imerg'])
        expts_ac = {expt: xr.load_dataset(self.inputs[expt]) for expt in EXPT_SIM}

        fig, axes = plt.subplots(4, 4, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
        fig.set_size_inches((24, 12))
        im = axes[0, 0].pcolormesh(imerg_ac.longitude, imerg_ac.latitude, imerg_ac.imerg_autocorr, vmin=0, vmax=1)
        # plt.colorbar(im, ax=axes[0])
        axes[0, 0].coastlines()

        for ax, expt_ac in zip(axes[0, 1:], expts_ac.values()):
            im = ax.pcolormesh(expt_ac.longitude, expt_ac.latitude, expt_ac.pflux_autocorr.mean(dim='ens_mem'), vmin=0, vmax=1)
            # plt.colorbar(im, ax=ax)
            ax.coastlines()

        plt.colorbar(im, ax=axes[0, -1])

        for ax, expt_ac in zip(axes[1, 1:], expts_ac.values()):
            im = ax.pcolormesh(expt_ac.longitude, expt_ac.latitude, expt_ac.pflux_autocorr.mean(dim='ens_mem') - imerg_ac.imerg_autocorr, vmin=-.4, vmax=.4, cmap='bwr')
            # plt.colorbar(im, ax=ax)
            ax.coastlines()

        plt.colorbar(im, ax=axes[1, -1])

        for ax, expt_ac in zip(axes[2, 2:], list(expts_ac.values())[1:]):
            im = ax.pcolormesh(expt_ac.longitude, expt_ac.latitude, expt_ac.pflux_autocorr.mean(dim='ens_mem') - expts_ac['Control'].pflux_autocorr.mean(dim='ens_mem'), vmin=-.2, vmax=.2, cmap='bwr')
            # plt.colorbar(im, ax=ax)
            ax.coastlines()

        plt.colorbar(im, ax=axes[2, -1])


        ax = axes[3, 3]
        im = ax.pcolormesh(expt_ac.longitude, expt_ac.latitude, expts_ac['vanillaMCSP'].pflux_autocorr.mean(dim='ens_mem') - expts_ac['STOCH-PRIME-MCSP'].pflux_autocorr.mean(dim='ens_mem'), vmin=-.2, vmax=.2, cmap='bwr')
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

        plt.savefig(self.outputs['fig'])

