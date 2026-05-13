from itertools import product

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import xarray as xr


class ASoPlite:
    """Re-implementation of the Analysis Scales of Precipitation (ASoP) toolkit.
    
    * Klingaman et al., 2017 (KMM17): https://doi.org/10.5194/gmd-10-57-2017
    * Aim to match their colour scales exactly (for better or worse)
    """
    def __init__(self, da, time_mean, biperiodic='neither', aspect=1.5, **user_config):
        """xarray DataArray (da) must have 3 dims, time, latitude, longitude, in any order (including lon/lat)

        time_mean is user-supplied averaging, e.g., hourly, 3-hourly.
        aspect is ratio of x-dir to y-dir grid spacing (at tropics) (defaults to UM N216 grid)
        Any config can be overridden with keyword args."""
        if 'lon' in da.coords and 'lat' in da.coords:
            da = da.rename(lon='longitude', lat='latitude')
        if da.dims != ('time', 'latitude', 'longitude'):
            da = da.transpose('time', 'latitude', 'longitude')
        assert da.dims == ('time', 'latitude', 'longitude'), 'DataArray has wrong coords/dims'

        self.da = da
        self.time_mean = time_mean
        self.biperiodic = biperiodic
        # N216 grid means spacing of:
        # 360 / (216 * 2) == 0.833deg in x-dir
        # 180 / (216 * 1.5) == 0.555deg in y-dir
        # dx == 1.5 * dy.
        self.aspect = aspect
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
        """Combine all numpy arrays into an easy-to-access dataset."""
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
        """Utility of dataset has been saved elsewhere."""
        self.ds = ds

    def print_config(self):
        """Print config settings."""
        for k, v in self.config.items():
            print(f'{k}: {v}')

    def calc_all(self):
        """Do all calculations."""
        self.calc_fractional_contrib()
        self.calc_precip_prob_matrix()
        self.calc_7x7_spat_corr()
        self.calc_7x7_spat_temp_corr()
        self.gen_ds()

    def plot_all(self):
        """Plot all figures."""
        self.plot_fractional_contrib()
        self.plot_precip_prob_matrix()
        self.plot_7x7_spat_corr()
        self.plot_7x7_spat_temp_corr()

    def calc_fractional_contrib(self):
        """Calc the fractional contrib, as in KMM17, Fig 1."""
        da = self.da.load()
        self.fractional_contrib = []

        thresh = self.fractional_contrib_thresh_mmpday / 24
        dig_data = np.digitize(da.values, thresh)
        for i in range(1, 5):
            self.fractional_contrib.append((da * (dig_data == i)).sum(dim='time').values / da.sum(dim='time').values)

    def plot_fractional_contrib(self, axes=None, colorbar=True):
        """Plot the fractional contrib, as in KMM17, Fig 1."""
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

        if colorbar:
            plt.colorbar(im, ax=axes, orientation='horizontal', extend='min', label='fractional contribution')
        # Return final im in case caller wants to plot colorbar.
        return im

    def calc_precip_prob_matrix(self):
        """Calc the probability matrix, as in KMM17, Fig 2a."""
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
        """Plot the probability matrix, as in KMM17, Fig 2a."""
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
        """Calc the 7x7 spatial correlation matrix, as in KMM17, Fig 2c."""

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
        """Plot the 7x7 spatial correlation matrix, as in KMM17, Fig 2c."""
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
        """Calc the 7x7 spatio-temporal correlation matrix, as in KMM17, Fig 2e."""
        da = self.da.load()
        biperiodic = self.biperiodic
        self.spat_temp_corr = np.zeros((5, 7, 7))
        for i, j, k in product(range(7), range(7), range(5)):
            print(i, j, k)
            shift_lon = i - 3
            shift_lat = j - 3
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
        """Plot the 7x7 spatio-temporal correlation matrix, as in KMM17, Fig 2e."""
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

        x = np.linspace(-3, 3, 7) * self.aspect # Corrects for grid *at equator*.
        y = np.linspace(-3, 3, 7)
        X, Y = np.meshgrid(x, y)
        bins = np.array([0, 0.5, 1.5, 2.5, 3.5])
        d = np.digitize(np.sqrt(X**2 + Y**2), bins)

        spat_temp_corr2 = np.zeros((5, 4))
        for i in range(5):
            for j in range(1, 5):
                spat_temp_corr2[i, j - 1] = self.ds.spat_temp_corr.values[i, :, :][d == j].mean()

        im = ax.imshow(spat_temp_corr2, origin='lower', cmap=cmap, norm=norm, aspect=1 / 1.3)
        ax.set_xticks(range(4))
        ax.set_yticks(range(5))
        for i in range(5):
            for j in range(4):
                val = spat_temp_corr2[i, j]
                text = f'{val:.2f}'
                c = 'k' if val <= 0.35 else 'white'
                ax.text(j, i, text, ha='center', va='center', color=c)
        return im