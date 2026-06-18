# theta_e_remakefile.py
#
# Time-height theta_e over several study domains, 1-10 Sept 2020.
# ERA5 is fetched + cached per domain and compared against three MCS-PRIME
# model runs (one figure per domain).
#
# DAG (all per-domain rules matrixed over DOMAINS):
#   fetch_era5[domain]   (source: pull + cache area-mean t/q profile)
#       |
#   era5_theta_e[domain] (Td -> theta_e, standardised to (time, pressure))
#       |                  + external model theta_e.nc files
#       |
#   plot_filled[domain] / plot_contour[domain]  (2x2 panels, common time range)
#   plot_domains                                  (global map of all domains)
#
# Run everything:        uv run remake run theta_e_remakefile.py
# Download ERA5 only:     uv run remake run theta_e_remakefile.py -Q "rule == 'fetch_era5'"

import os

from remake import Remake, rule

rmk = Remake()

# --- config -----------------------------------------------------------------

# Study domains: lon/lat bounds (W and S negative); `dir` is the matching
# MCS-PRIME model output subdirectory. All domains have both ERA5 and model
# data.
DOMAINS = {
    'atlantic':       {'lon': (-26, -22), 'lat': (6, 10),  'dir': 'atlantic_26W-22W_6N-10N'},     # noqa: E501
    'bay_of_bengal':  {'lon': (84, 94),   'lat': (8, 18),  'dir': 'bay_of_bengal_84E-94E_8N-18N'},  # noqa: E501
    'philippine_sea': {'lon': (127, 137), 'lat': (7, 17),  'dir': 'philippine_sea_127E-137E_7N-17N'},  # noqa: E501
    'indian_ocean':   {'lon': (55, 85),   'lat': (-5, 5),  'dir': 'indian_ocean_55E-85E_5S-5N'},  # noqa: E501
}
DATES = ('2020-09-01', '2020-09-10')
PLEVELS = (100, 1000)                            # hPa, troposphere

MODEL_BASE = os.path.expanduser(
    '~/mirrors/jasmin/gws/ssde/j25b/mcs_prime/mmuetz/data/'
    'mcs_prime_output_remake3_run3/theta_e'
)
RUNS = ['Control', 'PRIME-MCSP', 'STOCH-PRIME-MCSP']
PANELS = ['ERA5'] + RUNS


def model_path(run, domain):
    return (f'{MODEL_BASE}/{run}/20200901T0000Z/'
            f'{DOMAINS[domain]["dir"]}/theta_e.nc')


def plot_inputs(domain):
    """ERA5 (built here) + the three external model files for this domain."""
    inputs = {'ERA5': f'data/theta_e/{domain}/ERA5.nc'}
    inputs.update({run: model_path(run, domain) for run in RUNS})
    return inputs


def load_clipped(inputs):
    """Open every theta_e field and clip to the common (overlapping) time range."""
    import xarray as xr

    fields = {name: xr.open_dataset(path).theta_e for name, path in inputs.items()}
    t0 = max(f.time.min().values for f in fields.values())
    t1 = min(f.time.max().values for f in fields.values())
    return {name: f.sel(time=slice(t0, t1)) for name, f in fields.items()}


# --- rules ------------------------------------------------------------------

@rule(
    outputs={'profile': 'data/era5/{domain}/era5_tq_profile.nc'},
    matrix={'domain': list(DOMAINS)},
    uses={'DOMAINS': DOMAINS, 'DATES': DATES, 'PLEVELS': PLEVELS},
)
def fetch_era5(outputs, domain):
    """Source rule: pull each domain box from Arraylake's time-chunked group
    and cache the area-averaged t/q profile so we never re-download."""
    import xarray as xr
    from arraylake import Client

    client = Client()
    client.login()  # requires free Earthmover account
    repo = client.get_repo('earthmover-public/era5')
    session = repo.readonly_session(branch='main')
    # "temporal" group is chunked along time -> cheap for a small box.
    ds = xr.open_zarr(session.store, group='pressure/temporal', chunks={})

    lon_min, lon_max = DOMAINS[domain]['lon']
    if float(ds.longitude.max()) > 180:          # dataset uses 0-360 longitudes
        lon_min, lon_max = lon_min % 360, lon_max % 360
    lat_lo, lat_hi = DOMAINS[domain]['lat']
    p_lo, p_hi = PLEVELS

    box = (
        ds[['t', 'q']]
        .sel(valid_time=slice(*DATES))
        .sortby('latitude')
        .sortby('longitude')
        .sortby('pressure_level')
        .sel(latitude=slice(lat_lo, lat_hi), longitude=slice(lon_min, lon_max))
        .sel(pressure_level=slice(p_lo, p_hi))
    )
    prof = box.mean(['latitude', 'longitude']).load()
    prof.to_netcdf(outputs['profile'])


@rule(
    inputs=fetch_era5.outputs,
    outputs={'theta_e': 'data/theta_e/{domain}/ERA5.nc'},
    matrix=fetch_era5.matrix,
    depends_on=[fetch_era5],
)
def era5_theta_e(inputs, outputs, domain):
    """Derive dewpoint from q, then theta_e; standardise dim names to match
    the model files (time, pressure)."""
    import metpy.calc as mpcalc
    import xarray as xr

    prof = xr.open_dataset(inputs['profile'])
    prof.t.attrs['units'] = 'kelvin'
    prof.q.attrs['units'] = 'kg/kg'
    prof.pressure_level.attrs['units'] = 'hPa'

    td = mpcalc.dewpoint_from_specific_humidity(prof.pressure_level, prof.q)
    theta_e = mpcalc.equivalent_potential_temperature(prof.pressure_level, prof.t, td)
    theta_e = theta_e.metpy.dequantify().rename('theta_e')
    theta_e = theta_e.rename({'valid_time': 'time', 'pressure_level': 'pressure'})
    theta_e.to_netcdf(outputs['theta_e'])


@rule(
    inputs=plot_inputs,
    outputs={'fig': 'figs/theta_e_filled_{domain}.png'},
    matrix={'domain': list(DOMAINS)},
    depends_on=[era5_theta_e],
    uses={'PANELS': PANELS, 'load_clipped': load_clipped},
)
def plot_filled(inputs, outputs, domain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fields = load_clipped(inputs)

    all_vals = np.concatenate([fields[n].values.ravel() for n in PANELS])
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])
    flevels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, PANELS):
        cf = fields[name].plot.contourf(
            x='time', y='pressure', levels=flevels, cmap='turbo',
            yincrease=False, ax=ax, add_colorbar=False, extend='both',
        )
        ax.set_title(name)
        ax.set_xlabel('')
        ax.set_ylabel('Pressure (hPa)')

    fig.colorbar(cf, ax=axes, label=r'$\theta_e$ (K)', shrink=0.85)
    fig.suptitle(rf"Equivalent potential temperature $\theta_e$ — {domain.replace('_', ' ')}")
    fig.autofmt_xdate()
    fig.savefig(outputs['fig'], dpi=120, bbox_inches='tight')
    plt.close(fig)


@rule(
    inputs=plot_inputs,
    outputs={'fig': 'figs/theta_e_contour_{domain}.png'},
    matrix={'domain': list(DOMAINS)},
    depends_on=[era5_theta_e],
    uses={'PANELS': PANELS, 'load_clipped': load_clipped},
)
def plot_contour(inputs, outputs, domain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fields = load_clipped(inputs)

    # 330-350 K every 2 K, then 360-400 K every 10 K; 332 & 340 K thicker.
    levels = np.concatenate([np.arange(330, 351, 2), np.arange(360, 401, 10)])
    bold = {332, 340}
    linewidths = [1.6 if lev in bold else 0.6 for lev in levels]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, PANELS):
        cs = fields[name].plot.contour(
            x='time', y='pressure', levels=levels, colors='black',
            linewidths=linewidths, yincrease=False, ax=ax,
        )
        ax.clabel(cs, levels=levels[levels % 10 == 0], fmt='%d', fontsize=8)
        ax.set_title(name)
        ax.set_xlabel('')
        ax.set_ylabel('Pressure (hPa)')

    fig.suptitle(rf"Equivalent potential temperature $\theta_e$ (K) — {domain.replace('_', ' ')}")
    fig.autofmt_xdate()
    fig.savefig(outputs['fig'], dpi=120, bbox_inches='tight')
    plt.close(fig)


@rule(
    inputs=plot_inputs,
    outputs={'fig': 'figs/theta_e_mean_profile_{domain}.png'},
    matrix={'domain': list(DOMAINS)},
    depends_on=[era5_theta_e],
    uses={'PANELS': PANELS, 'load_clipped': load_clipped},
)
def plot_mean_profile(inputs, outputs, domain):
    """Single panel: time-mean theta_e profile for each data source."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fields = load_clipped(inputs)

    fig, ax = plt.subplots(figsize=(6, 7))
    for name in PANELS:
        prof = fields[name].mean('time')
        color = 'black' if name == 'ERA5' else None
        ax.plot(prof, prof.pressure, marker='o', markersize=3, color=color,
                label=name)

    ax.invert_yaxis()                               # pressure decreasing upward
    ax.set_xlabel(r'$\theta_e$ (K)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title(rf"Mean $\theta_e$ profile — {domain.replace('_', ' ')}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(outputs['fig'], dpi=120, bbox_inches='tight')
    plt.close(fig)


@rule(
    outputs={'fig': 'figs/study_domains.png'},
    uses={'DOMAINS': DOMAINS},
)
def plot_domains(outputs):
    """Source rule: global map with a shaded-relief background marking all
    study domains."""
    import matplotlib
    matplotlib.use('Agg')
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()                                  # shaded-relief background
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    for name, box in DOMAINS.items():
        lon_w, lon_e = box['lon']
        lat_s, lat_n = box['lat']
        ax.add_patch(mpatches.Rectangle(
            (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
            edgecolor='red', facecolor='none', linewidth=2,
            transform=ccrs.PlateCarree(), zorder=5,
        ))
        ax.text(lon_w, lat_n + 2, name.replace('_', ' '), color='red',
                fontweight='bold', fontsize=9, transform=ccrs.PlateCarree(),
                zorder=6)

    ax.set_title('Study domains')
    fig.savefig(outputs['fig'], dpi=120, bbox_inches='tight')
    plt.close(fig)


def report_inputs():
    """The domain map plus per-domain contour and mean-profile figures."""
    inputs = {'map': 'figs/study_domains.png'}
    for d in DOMAINS:
        inputs[f'contour_{d}'] = f'figs/theta_e_contour_{d}.png'
        inputs[f'profile_{d}'] = f'figs/theta_e_mean_profile_{d}.png'
    return inputs


@rule(
    inputs=report_inputs,
    outputs={'pdf': '_build/theta_e_report.pdf'},
    depends_on=[plot_domains, plot_contour, plot_mean_profile],
    uses={'DOMAINS': DOMAINS},
)
def build_report(inputs, outputs):
    """Assemble the figures into a LaTeX report and compile with latexmk
    into _build/ (no aux files left in the project root)."""
    import subprocess
    from pathlib import Path

    build = Path(outputs['pdf']).parent
    build.mkdir(parents=True, exist_ok=True)

    def img(key):
        return Path(inputs[key]).resolve().as_posix()

    lines = [
        r'\documentclass{article}',
        r'\usepackage[margin=2cm]{geometry}',
        r'\usepackage{graphicx}',
        r'\pagestyle{empty}',
        r'\title{$\theta_e$ in obs and models}',
        r'\date{}',
        r'\begin{document}',
        r'\maketitle',
        r'\vfill',
        r'\begin{center}',
        rf'\includegraphics[width=\linewidth]{{{img("map")}}}\par',
        r'\smallskip',
        r'\small Study domains.',
        r'\end{center}',
        r'\vfill',
        r'\clearpage',
    ]
    for d in DOMAINS:
        name = d.replace('_', ' ').title()
        lines += [
            rf'\section*{{{name}}}',
            r'\begin{center}',
            rf'\includegraphics[width=\linewidth]{{{img(f"contour_{d}")}}}\par',
            r'\smallskip',
            r'\small Time--height $\theta_e$: ERA5 and three model runs.',
            r'\end{center}',
            r'\vfill',
            r'\begin{center}',
            rf'\includegraphics[width=0.5\linewidth]{{{img(f"profile_{d}")}}}\par',
            r'\smallskip',
            r'\small Domain-mean $\theta_e$ profile.',
            r'\end{center}',
            r'\clearpage',
        ]
    lines.append(r'\end{document}')

    tex = build / 'theta_e_report.tex'
    tex.write_text('\n'.join(lines))

    # Run in build/ so every aux file stays there, not in the project root.
    subprocess.run(
        ['latexmk', '-pdf', '-interaction=nonstopmode', tex.name],
        cwd=build, check=True,
    )


rmk.rules_from_current_module()
