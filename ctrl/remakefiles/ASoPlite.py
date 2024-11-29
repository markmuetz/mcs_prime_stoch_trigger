from itertools import product

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
import shapely
import xarray as xr

from remake import Remake, Rule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'

EXPTS = {
    'ctrl': 'u-di727',
    'vanillaMCSP': 'u-di728',
    'stochMCSP': 'u-dg135',
}


class ASoPlite:
    def __init__(self, da, time_mean, biperiodic='neither', **user_config):
        """xarray DataArray (da) must have 3 dims, time, latitude, longitude, in any order (including lon/lat)

        time_mean is user-supplied averaging, e.g., hourly, 3-hourly.
        Any config can be overridden with keyword args."""
        if 'lon' in da.coords and 'lat' in da.coords:
            da = da.rename(lon='longitude', lat='latitude')
        if da.dims != ('time', 'latitude', 'longitude'):
            da = da.transpose('time', 'latitude', 'longitude')
        assert da.dims == ('time', 'latitude', 'longitude'), 'DataArray has wrong coords/dims'

        self.da = da
        self.time_mean = time_mean
        self.biperiodic = biperiodic
        self.config = {
            'precip_prob_matrix_bins_mmpday': np.array(
                [0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 40, 60, 90, 130, 180, 100000]
            ),
            'precip_prob_matrix_boundaries': [
                1e-5,
                2e-5,
                3e-5,
                4e-5,
                7e-5,
                1e-4,
                2e-4,
                4e-4,
                7e-4,
                1e-3,
                2e-3,
                4e-3,
                7e-3,
                1e-2,
                7e-2,
                1e-1,
                1,
            ],
            'fractional_contrib_thresh_mmpday': np.array([0.005, 10, 50, 100]),
        }
        self.config.update(user_config)

        for k, v in self.config.items():
            setattr(self, k, v)

    def gen_ds(self):
        ds = xr.Dataset()
        ds['fractional_contrib'] = xr.DataArray(
            self.fractional_contrib,
            dims=('frac_contrip_precip_bin_lower', 'latitude', 'longitude'),
            coords=dict(
                frac_contrip_precip_bin_lower=(
                    'frac_contrip_precip_bin_lower',
                    self.fractional_contrib_thresh_mmpday,
                    {'units': 'mm day-1'},
                ),
                latitude=self.da.latitude,
                longitude=self.da.longitude,
            ),
        )

        ds['precip_prob_hist'] = xr.DataArray(
            self.precip_prob_hist,
            dims='precip_prob_bin_lower',
            coords=dict(
                precip_prob_bin_lower=(
                    'precip_prob_bin_lower',
                    self.precip_prob_matrix_bins_mmpday[:-1],
                    {'units': 'mm day-1'},
                )
            ),
        )

        ds['precip_prob_matrix'] = xr.DataArray(
            self.precip_prob_matrix,
            dims=('precip_prob_bin_lower_t0', 'precip_prob_bin_lower_t1'),
            coords=dict(
                precip_prob_bin_lower_t0=(
                    'precip_prob_bin_lower_t0',
                    self.precip_prob_matrix_bins_mmpday[1:],
                    {'units': 'mm day-1'},
                ),
                precip_prob_bin_lower_t1=(
                    'precip_prob_bin_lower_t1',
                    self.precip_prob_matrix_bins_mmpday[1:],
                    {'units': 'mm day-1'},
                ),
            ),
        )

        ds['spat_corr'] = xr.DataArray(
            self.spat_corr,
            dims=('offset_x', 'offset_y'),
            coords=dict(
                offset_x=('offset_x', range(-3, 4), {'units': 'delta x'}),
                offset_y=('offset_y', range(-3, 4), {'units': 'delta y'}),
            ),
        )

        ds['spat_temp_corr'] = xr.DataArray(
            self.spat_temp_corr,
            dims=('offset_t', 'offset_x', 'offset_y'),
            coords=dict(
                offset_t=('offset_t', range(5), {'units': 'delta t'}),
                offset_x=('offset_x', range(-3, 4), {'units': 'delta x'}),
                offset_y=('offset_y', range(-3, 4), {'units': 'delta y'}),
            ),
        )
        self.ds = ds
        return self.ds

    def load_ds(self, ds):
        self.ds = ds

    def print_config(self):
        for k, v in self.config.items():
            print(f'{k}: {v}')

    def calc_all(self):
        self.calc_fractional_contrib()
        self.calc_precip_prob_matrix()
        self.calc_7x7_spat_corr()
        self.calc_7x7_spat_temp_corr()
        self.gen_ds()

    def plot_all(self):
        self.plot_fractional_contrib()
        self.plot_precip_prob_matrix()
        self.plot_7x7_spat_corr()
        self.plot_7x7_spat_temp_corr()

    def calc_fractional_contrib(self):
        da = self.da.load()
        self.fractional_contrib = []

        thresh = self.fractional_contrib_thresh_mmpday / 24
        dig_data = np.digitize(da.values, thresh)
        for i in range(1, 5):
            self.fractional_contrib.append((da * (dig_data == i)).sum(dim='time').values / da.sum(dim='time').values)

    def plot_fractional_contrib(self, axes=None):
        nplots = len(self.fractional_contrib_thresh_mmpday)
        if axes is None:
            fig, axes = plt.subplots(nplots, 1, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
        thresh = self.fractional_contrib_thresh_mmpday / 24

        html_colours = [
            '#2166ac',
            '#4393c3',
            '#92c5de',
            '#d1e5f0',
            '#f7f7f7',
            '#fddbc7',
            '#f4a582',
            '#d6604d',
            '#b2182b',
            '#ec7014',
            '#fe9929',
            '#fec44f',
            '#fee391',
            '#fff7bc',
        ]
        under_colour = '#2166ac'
        over_colour = '#fff7bc'

        cmap = mcolors.ListedColormap(html_colours[1:-1])
        cmap.set_under(under_colour)
        cmap.set_over(over_colour)
        norm = mcolors.BoundaryNorm([0.05, 0.1, 0.15, 0.2, 0.25, 0.325, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], cmap.N)

        for i in range(nplots):
            ax = axes[i]
            ax.coastlines(lw=2)
            pdata = self.ds.fractional_contrib[i]
            im = ax.pcolormesh(pdata.longitude, pdata.latitude, pdata, norm=norm, cmap=cmap)
            if i < nplots - 1:
                t0 = thresh[i] * 24
                t1 = thresh[i + 1] * 24
                if i == 0:
                    ax.set_title(f'{self.time_mean} events {t0:.3f}–{t1:.0f} mm day$^{{-1}}$')
                else:
                    ax.set_title(f'{self.time_mean} events {t0:.0f}–{t1:.0f} mm day$^{{-1}}$')
            else:
                t = thresh[-1] * 24
                ax.set_title(f'{self.time_mean} events >{t:.0f} mm day$^{{-1}}$')

        plt.colorbar(im, ax=ax, orientation='horizontal', extend='min', label='fractional contribution')

    def calc_precip_prob_matrix(self):
        da = self.da.load()
        data = da.values

        bins = self.precip_prob_matrix_bins_mmpday / 24
        self.precip_prob_hist = np.histogram(data.flatten(), bins=bins)[0]
        self.precip_prob_hist = self.precip_prob_hist / self.precip_prob_hist.sum()

        dbins = bins[1:-1]
        dig_data = np.digitize(data, bins=dbins)

        self.precip_prob_matrix = np.zeros((len(dbins) + 1, len(dbins) + 1))
        for p in zip(dig_data[:-1].flatten(), dig_data[1:].flatten()):
            self.precip_prob_matrix[p] += 1
        self.precip_prob_matrix = self.precip_prob_matrix / self.precip_prob_matrix.sum()

    def plot_precip_prob_matrix(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10), layout='constrained')
        boundaries = self.precip_prob_matrix_boundaries
        norm = mcolors.BoundaryNorm(boundaries, 256)

        # If using imshow, this is bizarly sensitive to the *exact* figsize. Very weird.
        ax2 = ax.twinx()
        print(self.ds.precip_prob_hist)
        ax2.plot(
            np.arange(0.5, len(self.ds.precip_prob_hist.values) + 0.5),
            self.ds.precip_prob_hist.values,
            ls='--',
            marker='o',
            c='k',
        )
        ax2.set_yscale('log')

        # Bad things happen with alignment of ax if you use imshow.
        # im = ax.imshow(probs, origin='lower', norm=norm, cmap='viridis_r')
        im = ax.pcolormesh(self.ds.precip_prob_matrix, norm=norm, cmap='viridis_r')

        def cbar_fmt_tick(v, pos):
            return f'{v:.0e}'

        plt.colorbar(
            im,
            ax=ax,
            ticks=boundaries[:-1],
            format=FuncFormatter(cbar_fmt_tick),
            boundaries=np.linspace(0, 1, len(boundaries) - 1),
        )

        dbins_mmpday = self.precip_prob_matrix_bins_mmpday[1:-1]
        ticks = np.arange(len(dbins_mmpday) + 2)
        ticklabels = ['<1'] + [f'{b:.0f}' for b in dbins_mmpday] + ['>180']
        ax.set_xticks(ticks, ticklabels)
        ax.set_yticks(ticks, ticklabels)

        ax2.set_ylim((1e-4, 1e0))

        ax.set_xlabel('precip. at time t (mm day$^{-1}$)')
        ax.set_ylabel('precip. at time t+1 (mm day$^{-1}$)')
        ax2.set_ylabel('probability of precip. in bin')

    def calc_7x7_spat_corr(self):
        da = self.da.load()
        if self.biperiodic == 'neither':
            xyslice = (None, slice(3, -3), slice(3, -3))
        elif self.biperiodic == 'x':
            xyslice = (None, slice(3, -3), None)
        elif self.biperiodic == 'y':
            xyslice = (None, None, slice(3, -3))
        elif self.biperiodic == 'both':
            xyslice = (None, slice(3, -3), slice(3, -3))

        values = da.values
        values[np.isnan(values)] = 0
        values_flat = values[xyslice].flatten()
        self.spat_corr = np.zeros((7, 7))
        for i, j in product(range(7), range(7)):
            print(i, j)
            shift_lon = i - 3
            shift_lat = j - 3
            self.spat_corr[j, i] = scipy.stats.pearsonr(
                values_flat, np.roll(np.roll(values, shift_lon, axis=-1), shift_lat, axis=-2)[xyslice].flatten()
            )[0]
        return self.spat_corr

    def plot_7x7_spat_corr(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        spat_corr = self.ds.spat_corr.values
        html_colours = [
            '#e8e4e7',
            '#fecd61',
            '#d5c24e',
            '#9ec039',
            '#55b545',
            '#399e69',
            '#398b79',
            '#37767c',
            '#2a6089',
            '#2b449d',
            '#361f71',
        ]
        under_colour = '#e8e4e7'
        over_colour = '#361f71'

        cmap = mcolors.ListedColormap(html_colours[1:-1])
        cmap.set_under(under_colour)
        cmap.set_over(over_colour)
        norm = mcolors.BoundaryNorm([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], cmap.N)

        df = pd.DataFrame(spat_corr, columns=[f'{v}' for v in range(-3, 4)])[::-1]
        df.index = range(3, -4, -1)
        sns.heatmap(df, ax=ax, annot=True, norm=norm, cmap=cmap)

    def calc_7x7_spat_temp_corr(self):
        da = self.da.load()
        biperiodic = self.biperiodic
        self.spat_temp_corr = np.zeros((5, 7, 7))
        for i, j, k in product(range(7), range(7), range(5)):
            print(i, j, k)
            shift_lon = i - 3
            shift_lat = j - 3
            shift_time = k
            if k == 0:
                if biperiodic == 'neither':
                    xyslice1 = (None, slice(3, -3), slice(3, -3))
                elif biperiodic == 'x':
                    xyslice1 = (None, slice(3, -3), None)
                elif biperiodic == 'y':
                    xyslice1 = (None, None, slice(3, -3))
                elif biperiodic == 'both':
                    xyslice1 = (None, slice(3, -3), slice(3, -3))
                xyslice2 = xyslice1
            else:
                if biperiodic == 'neither':
                    xyslice1 = (slice(k, None), slice(3, -3), slice(3, -3))
                    xyslice2 = (slice(None, -k), slice(3, -3), slice(3, -3))
                elif biperiodic == 'x':
                    xyslice1 = (slice(k, None), slice(3, -3), None)
                    xyslice2 = (slice(None, -k), slice(3, -3), None)
                elif biperiodic == 'y':
                    xyslice1 = (slice(k, None), None, slice(3, -3))
                    xyslice2 = (slice(None, -k), None, slice(3, -3))
                elif biperiodic == 'both':
                    xyslice1 = (slice(k, None), slice(3, -3), slice(3, -3))
                    xyslice2 = (slice(None, -k), slice(3, -3), slice(3, -3))

            self.spat_temp_corr[k, j, i] = scipy.stats.pearsonr(
                da.values[xyslice1].flatten(),
                np.roll(np.roll(da.values, shift_lon, axis=-1), shift_lat, axis=-2)[xyslice2].flatten(),
            )[0]
        return self.spat_temp_corr

    def plot_7x7_spat_temp_corr(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        html_colours = [
            '#e8e4e7',
            '#fecd61',
            '#d5c24e',
            '#9ec039',
            '#55b545',
            '#399e69',
            '#398b79',
            '#37767c',
            '#2a6089',
            '#2b449d',
            '#361f71',
        ]
        under_colour = '#e8e4e7'
        over_colour = '#361f71'

        cmap = mcolors.ListedColormap(html_colours[1:-1])
        cmap.set_under(under_colour)
        cmap.set_over(over_colour)
        norm = mcolors.BoundaryNorm([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], cmap.N)

        x = np.linspace(-3, 3, 7)
        y = np.linspace(-3, 3, 7)
        X, Y = np.meshgrid(x, y)
        bins = np.array([0, 0.5, 1.5, 2.5, 3.5])
        d = np.digitize(np.sqrt(X**2 + Y**2), bins)

        spat_temp_corr2 = np.zeros((5, 4))
        for i in range(5):
            for j in range(1, 5):
                spat_temp_corr2[i, j - 1] = self.ds.spat_temp_corr.values[i, :, :][d == j].mean()
        sns.heatmap(pd.DataFrame(spat_temp_corr2)[::-1], ax=ax, annot=True, norm=norm, cmap=cmap)


rmk = Remake({})

REGIONS = {
    'eq_warm_pool': (60, 160, -10, 10),
    'eq_band': (0, 360, -10, 10),
    'indian_ocean': (50, 100, -10, 10),
    'india': (70, 90, 10, 30),
    'west_pacific': (110, 170, 5, 30),
    'china': (100, 120, 22, 32),
    'us': (245, 275, 32, 48),
}


class ASoPN216regional(Rule):
    rule_matrix = {
        'expt': ['imerg', 'ctrl', 'vanillaMCSP', 'stochMCSP'],
        'region': list(REGIONS),
        'coarsen_time': ['hourly', '3-hourly'],
    }

    @staticmethod
    def rule_inputs(expt, region, coarsen_time):
        if expt == 'imerg':
            inputs = {
                'precip': (
                    cu.PATHS['outdir']
                    / 'imerg_processed/N216grid/2020-07-01_04:00:00-2020-07-11_03:00:00/'
                    / '3B-HHR.MS.MRG.3IMERG.2020-07-01_04:00:00-2020-07-11_03:00:00.hourly.V07B.nc'
                )
            }
        else:
            suite = EXPTS[expt]
            inputs = {'precip': SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'}
        return inputs

    @staticmethod
    def rule_outputs(expt, region, coarsen_time):
        return {'output': cu.PATHS['outdir'] / 'ASoP' / 'dev' / f'asop.{expt}.{region}.{coarsen_time}.nc'}

    def rule_run(self):
        reg_extent = REGIONS[self.region]
        lat_lon_sel = dict(longitude=slice(reg_extent[0], reg_extent[1]), latitude=slice(reg_extent[2], reg_extent[3]))

        if self.expt == 'imerg':
            da_precip = xr.open_dataarray(self.inputs['precip']).sel(**lat_lon_sel)
        else:
            da_precip = xr.open_dataarray(self.inputs['precip']).sel(ens_mem=1, **lat_lon_sel)
            da_precip.values *= 3600
            da_precip.attrs['units'] = 'mm h-1'

        if self.coarsen_time == '3-hourly':
            da_precip = da_precip.coarsen(time=3).mean()
        da_precip = da_precip.load()

        asop = ASoPlite(da_precip, self.coarsen_time)
        asop.calc_all()
        cu.to_netcdf_tmp_then_copy(asop.ds, self.outputs['output'])


class PlotRegions(Rule):
    @staticmethod
    def rule_inputs():
        return {}

    @staticmethod
    def rule_outputs():
        fig_asop_dir = cu.PATHS['figdir'] / 'ASoP' / 'dev'
        return {'asop_regs': fig_asop_dir / f'asop.regions.png'}

    def rule_run(self):
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
        plt.savefig(self.outputs['asop_regs'])


class PlotASoPN216regional(Rule):
    rule_matrix = {
        'region': list(REGIONS),
        'coarsen_time': ['hourly', '3-hourly'],
    }

    @staticmethod
    def rule_inputs(region, coarsen_time):
        inputs = {}
        for expt in ['imerg', 'ctrl', 'vanillaMCSP', 'stochMCSP']:
            if expt == 'imerg':
                inputs[f'precip_{expt}'] = (
                    cu.PATHS['outdir']
                    / 'imerg_processed/N216grid/2020-07-01_04:00:00-2020-07-11_03:00:00/'
                    / '3B-HHR.MS.MRG.3IMERG.2020-07-01_04:00:00-2020-07-11_03:00:00.hourly.V07B.nc'
                )
            else:
                suite = EXPTS[expt]
                inputs[f'precip_{expt}'] = SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'
            inputs[f'asop_{expt}'] = ASoPN216regional.rule_outputs(expt, region, coarsen_time)['output']
        return inputs

    @staticmethod
    def rule_outputs(region, coarsen_time):
        fig_asop_dir = cu.PATHS['figdir'] / 'ASoP' / 'dev'
        return {
            'fractional_contrib': fig_asop_dir / f'asop.fractional_contrib.{region}.{coarsen_time}.png',
            'precip_prob_matrix': fig_asop_dir / f'asop.precip_prob_matrix.{region}.{coarsen_time}.png',
            '7x7_spat_corr': fig_asop_dir / f'asop.7x7_spat_corr.{region}.{coarsen_time}.png',
            '7x7_spat_temp_corr': fig_asop_dir / f'asop.7x7_spat_temp_corr.{region}.{coarsen_time}.png',
        }

    def rule_run(self):
        reg_extent = REGIONS[self.region]
        lat_lon_sel = dict(longitude=slice(reg_extent[0], reg_extent[1]), latitude=slice(reg_extent[2], reg_extent[3]))

        asops = {}
        for expt in ['imerg', 'ctrl', 'vanillaMCSP', 'stochMCSP']:
            if expt == 'imerg':
                da_precip = xr.open_dataarray(self.inputs[f'precip_{expt}']).sel(**lat_lon_sel)
            else:
                da_precip = xr.open_dataarray(self.inputs[f'precip_{expt}']).sel(ens_mem=1, **lat_lon_sel)
                da_precip.values *= 3600
                da_precip.attrs['units'] = 'mm h-1'

            if self.coarsen_time == '3-hourly':
                da_precip = da_precip.coarsen(time=3).mean()
            asops[expt] = ASoPlite(da_precip, self.coarsen_time)
            asops[expt].load_ds(xr.open_dataset(self.inputs[f'asop_{expt}']))

        fig, axes = plt.subplots(4, 4, layout='constrained', subplot_kw={'projection': ccrs.PlateCarree()})
        fig.set_size_inches(20, 6)
        for axcol, (expt, asop) in zip(axes.T, asops.items()):
            asop.plot_fractional_contrib(axes=axcol)
        plt.savefig(self.outputs['fractional_contrib'])

        fig, axes = plt.subplots(2, 2, layout='constrained')
        fig.set_size_inches(16, 12)
        for ax, (expt, asop) in zip(axes.flatten(), asops.items()):
            asop.plot_precip_prob_matrix(ax=ax)
            fig = plt.gcf()
            ax.set_title(expt)
        plt.savefig(self.outputs['precip_prob_matrix'])

        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(12, 8)
        for ax, (expt, asop) in zip(axes.flatten(), asops.items()):
            asop.plot_7x7_spat_corr(ax=ax)
            ax.set_title(expt)
        plt.savefig(self.outputs['7x7_spat_corr'])

        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(12, 8)
        for ax, (expt, asop) in zip(axes.flatten(), asops.items()):
            asop.plot_7x7_spat_temp_corr(ax=ax)
            ax.set_title(expt)
        plt.savefig(self.outputs['7x7_spat_temp_corr'])
