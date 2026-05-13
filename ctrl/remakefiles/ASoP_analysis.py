import string

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import shapely
import xarray as xr

from remake import Remake, Rule

import proj_config as conf
import utils

from ASoPlite import ASoPlite


slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'standard', 'mem': 16000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

REGIONS = {
    'eq_warm_pool': (60, 160, -10, 10),
    'eq_band': (0, 360, -10, 10),
    'tropics': (0, 360, -30, 30),
    'indian_ocean': (50, 100, -10, 10),
    'india': (70, 90, 10, 30),
    'west_pacific': (110, 170, 5, 30),
    'china': (100, 120, 22, 32),
    'us': (245, 275, 32, 48),
}


class PlotRegions(Rule):
    """Plot the different regions on a global map."""
    @staticmethod
    def rule_inputs():
        return {}

    @staticmethod
    def rule_outputs():
        fig_asop_dir = conf.PATHS['figdir'] / 'ASoP'
        return {'asop_regs': fig_asop_dir / f'asop.regions.pdf'}

    @staticmethod
    def rule_run(inputs, outputs):
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained', subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray')
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 45))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 45))

        gl.top_labels = False
        gl.right_labels = False

        def f0_360to_m180_180(vs):
            return np.where(vs > 180, vs - 360, vs)

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        custom_lines = []

        for c, (bname, bcoords) in zip(colours, REGIONS.items()):
            if bname == 'eq_warm_pool':
                lw = 4
                zorder = 1
            else:
                lw = 2
                zorder = 0
            if bname == 'eq_band':
                minx, maxx, miny, maxy = (-180, 180, -10, 10)
            else:
                minx, maxx, miny, maxy = f0_360to_m180_180(np.array(bcoords))
            bpoints = ((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny))
            box = shapely.geometry.LinearRing(bpoints)
            # Add geometry for each nested grid size.
            ax.add_geometries([box], crs=ccrs.PlateCarree(), edgecolor=c, facecolor='none', lw=lw, zorder=zorder)
            custom_lines.append(Line2D([0], [0], color=c, lw=lw))
        ax.legend(custom_lines, REGIONS)
        plt.savefig(outputs['asop_regs'])


def open_precip(inputs, expt, region, coarsen_time):
    """Open precip DataArray, for given expt, region, and apply coarsening."""
    reg_extent = REGIONS[region]
    lat_lon_sel = dict(longitude=slice(reg_extent[0], reg_extent[1]), latitude=slice(reg_extent[2], reg_extent[3]))

    precip_inputs_values = [inputs[k] for k in inputs if k.startswith(f'precip_{expt}')]

    if expt.lower() == 'imerg':
        da_precip = xr.open_mfdataset(precip_inputs_values).__xarray_dataarray_variable__.sel(**lat_lon_sel)
    else:
        # Realization 1 (not 0, which is deterministic).
        da_precip = xr.open_mfdataset(precip_inputs_values).precipitation_flux.sel(realization=1, **lat_lon_sel)
        da_precip.values *= 3600  # convert to mm h-1
        da_precip.attrs['units'] = 'mm h-1'

    if coarsen_time == '3-hourly':
        da_precip = da_precip.coarsen(time=3).mean()
    return da_precip


class ASoPN216regional(Rule):
    """Use the ASoPlite class to calc the ASoP info for each region."""

    rule_matrix = {
        'expt': ['imerg', 'Control', 'PRIME-MCSP', 'STOCH-PRIME-MCSP'],
        'case': conf.CASES + ['all_cases'],
        'region': list(REGIONS),
        'coarsen_time': ['hourly', '3-hourly'],
    }

    @staticmethod
    def rule_inputs(expt, case, region, coarsen_time):
        inputs = {}
        if case == 'all_cases':
            cases = conf.CASES
        else:
            cases = [case]

        for case in cases:
            month = case[4:6]
            if expt == 'imerg':
                inputs[f'precip_{expt}_{case}'] = (
                    conf.PATHS['outdir']
                    / f'imerg_processed/N216grid/2020-{month}-01_04:00:00-2020-{month}-11_03:00:00/'
                    / f'N216.cons.3B-HHR.MS.MRG.3IMERG.2020-{month}-01_04:00:00-2020-{month}-11_03:00:00.hourly.V07B.nc'
                )
            else:
                suite = conf.EXPT_SIM[expt]
                inputs[f'precip_{expt}_{case}'] = conf.SIMDIR / f'{suite}/share/cycle/{case}/engl/um/englaa_pa.merged.{case}.{suite}.m01s05i216.1h-mean.nc'
        return inputs

    @staticmethod
    def rule_outputs(expt, case, region, coarsen_time):
        return {'output': conf.PATHS['outdir'] / 'ASoP' / expt / case / region / f'asop.{expt}.{case}.{region}.{coarsen_time}.nc'}

    @staticmethod
    def rule_run(inputs, outputs, expt, case, region, coarsen_time):
        print('Running', expt, case, region, coarsen_time)

        da_precip = open_precip(inputs, expt, region, coarsen_time)
        da_precip = da_precip.load()

        asop = ASoPlite(da_precip, coarsen_time)
        asop.calc_all()
        utils.to_netcdf_tmp_then_copy(asop.ds, outputs['output'])


class PlotASoPN216regional(Rule):
    """Plot the regional ASoP data using ASoPlite."""
    rule_matrix = {
        'region': list(REGIONS),
        'case': conf.CASES + ['all_cases'],
        'coarsen_time': ['hourly', '3-hourly'],
    }

    @staticmethod
    def rule_inputs(region, case, coarsen_time):
        inputs = {}
        for expt in ['imerg', 'Control', 'PRIME-MCSP', 'STOCH-PRIME-MCSP']:
            inputs.update(ASoPN216regional.rule_inputs(expt, case, region, coarsen_time))
            inputs[f'asop_{expt}'] = ASoPN216regional.rule_outputs(expt, case, region, coarsen_time)['output']
        return inputs

    @staticmethod
    def rule_outputs(region, case, coarsen_time):
        fig_asop_dir = conf.PATHS['figdir'] / 'ASoP'
        return {
            # This is a 15 MB file if I use .pdf.
            'fractional_contrib': fig_asop_dir / case / region / f'asop.fractional_contrib.{region}.{case}.{coarsen_time}.png',
            'precip_prob_matrix': fig_asop_dir / case / region / f'asop.precip_prob_matrix.{region}.{case}.{coarsen_time}.pdf',
            '7x7_spat_corr': fig_asop_dir / case / region / f'asop.7x7_spat_corr.{region}.{case}.{coarsen_time}.pdf',
            '7x7_spat_temp_corr': fig_asop_dir / case / region / f'asop.7x7_spat_temp_corr.{region}.{case}.{coarsen_time}.pdf',
        }

    @staticmethod
    def rule_run(inputs, outputs, region, case, coarsen_time):
        print('oh hai')
        asops = {}
        for expt in ['imerg', 'Control', 'PRIME-MCSP', 'STOCH-PRIME-MCSP']:
            da_precip = open_precip(inputs, expt, region, coarsen_time)
            asops[expt] = ASoPlite(da_precip, coarsen_time)
            asops[expt].load_ds(xr.open_dataset(inputs[f'asop_{expt}']))

        # fig, axes = plt.subplots(4, 4, layout='constrained', subplot_kw={'projection': ccrs.PlateCarree()}, dpi=600)
        # fig, axes = plt.subplots(9, 2, layout='constrained', subplot_kw={'projection': ccrs.PlateCarree()}, dpi=600)
        # fig.set_size_inches(10, 8)
        fig = plt.figure(layout='constrained', dpi=600)
        if region == 'eq_warm_pool':
            fig.set_size_inches(8, 10)
        else:
            fig.set_size_inches(8, 8)
        gs = fig.add_gridspec(10, 2, height_ratios=[1] * 4 + [0.3] + [1] * 4 + [0.3])

        axes = np.array([
            [fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree()) for c in range(2)]
            for r in [0, 1, 2, 3, 5, 6, 7, 8]
        ])
        cbar_ax = fig.add_subplot(gs[-1, :])  # spans both columns

        for axcol, (expt, asop) in zip([axes[:4, 0], axes[:4, 1], axes[4:, 0], axes[4:, 1]], asops.items()):
            print(f'i am in ur loopz {expt}')
            im = asop.plot_fractional_contrib(axes=axcol, colorbar=False)
            expt = expt.upper() if expt == 'imerg' else expt
            axcol[0].set_title(expt)
            axcol[1].set_title('')
            axcol[2].set_title('')
            axcol[3].set_title('')

        print('i maked one colorbar')
        plt.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='min', label='fractional contribution')

        for i, ax in enumerate([*axes[:4, 0], *axes[:4, 1], *axes[4:, 0], *axes[4:, 1]]):
            c = string.ascii_lowercase[i]
            ax.set_title(f'{c})', loc='left')
        for row in range(8):
            i = row % 4
            ax = axes[row, 0]
            if i < 3:
                t0 = asop.fractional_contrib_thresh_mmpday[i]
                t1 = asop.fractional_contrib_thresh_mmpday[i + 1]
                label = f'{t0:.0f}–{t1:.0f}\nmm day$^{{-1}}$'
            else:
                t0 = asop.fractional_contrib_thresh_mmpday[i]
                label = f'>{t0:.0f}\nmm day$^{{-1}}$'
            # ax.set_ylabel(label)
            # cartopy messes up set_ylabel - position manually.
            ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                    va='center', ha='center', rotation=90)

        print(f'i makde a picutre! {outputs["fractional_contrib"]}')
        plt.savefig(outputs['fractional_contrib'])

        fig, axes = plt.subplots(2, 2, layout='constrained')
        fig.set_size_inches(16, 12)
        for i, (ax, (expt, asop)) in enumerate(zip(axes.flatten(), asops.items())):
            asop.plot_precip_prob_matrix(ax=ax)
            expt = expt.upper() if expt == 'imerg' else expt
            ax.set_title(expt)
            c = string.ascii_lowercase[i]
            ax.set_title(f'{c})', loc='left')
        plt.savefig(outputs['precip_prob_matrix'])

        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(12, 8)
        for i, (ax, (expt, asop)) in enumerate(zip(axes.flatten(), asops.items())):
            asop.plot_7x7_spat_corr(ax=ax)
            expt = expt.upper() if expt == 'imerg' else expt
            ax.set_title(expt)
            c = string.ascii_lowercase[i]
            ax.set_title(f'{c})', loc='left')
        plt.savefig(outputs['7x7_spat_corr'])

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, layout='constrained')
        fig.set_size_inches(7, 5)
        for i, (ax, (expt, asop)) in enumerate(zip(axes.flatten(), asops.items())):
            im = asop.plot_7x7_spat_temp_corr(ax=ax)
            expt = expt.upper() if expt == 'imerg' else expt
            ax.set_title(expt)
            c = string.ascii_lowercase[i]
            ax.set_title(f'{c})', loc='left')
        plt.colorbar(im, ax=axes, orientation='vertical', label='Corr. with centre at lag=0')
        for ax in axes[:, 0]:
            if coarsen_time == '3-hourly':
                ax.set_ylabel('Lag (3 hourly)')
            else:
                ax.set_ylabel('Lag (hourly)')
        for ax in axes[-1, :]:
            ax.set_xlabel('$\\Delta x$ (approx. 75 km at equator)')
        plt.savefig(outputs['7x7_spat_temp_corr'])
