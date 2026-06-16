from pathlib import Path

from remake import util


# remake3 migration: outputs are redirected to a parallel *_remake3 tree so
# the existing remake2 outputs survive for the equivalence diff. Inputs
# (datadir, era5dir) stay shared - remake3 reads the same raw data.
# Flip this to False to write to the original (remake2) locations.
REMAKE3_EQUIVALENCE_TEST = True

PATHS = {
    'datadir': Path('/gws/ssde/j25b/mcs_prime/mmuetz/data/'),
    'outdir': Path('/gws/ssde/j25b/mcs_prime/mmuetz/data/mcs_prime_output'),
    'figdir': Path('/gws/ssde/j25b/mcs_prime/mmuetz/data/mcs_prime_figs/N216sims/prod'),
    # Checking this dir causing proc to hang.
    # 'era5dir': Path('/does/not/exist'),
    'era5dir': Path('/badc/ecmwf-era5'),
}

if REMAKE3_EQUIVALENCE_TEST:
    PATHS['outdir'] = Path('/gws/ssde/j25b/mcs_prime/mmuetz/data/mcs_prime_output_remake3')
    PATHS['figdir'] = Path('/gws/ssde/j25b/mcs_prime/mmuetz/data/mcs_prime_figs/N216sims/prod_remake3')

DATADIR = PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'
N_ENS_MEM = 10

EXPT_SIM = {
    'Control': 'u-di727',
    'PRIME-MCSP': 'u-di728',
    'STOCH-PRIME-MCSP': 'u-dg135',
}

CASES = [
    f'2020{m:02d}01T0000Z'
    for m in range(1, 13)
]

IMERG_FINAL_30MIN_DIR = DATADIR / 'GPM_IMERG_final/30min'
# These times exactly match the UM sims.
# UM_TIMES = pd.date_range('2020-07-01 04:00', '2020-07-11 03:00', freq='H')
UM_TIMES = {}
for case in CASES:
    m = case[4:6]
    UM_TIMES[case] = ((f'2020-{m}-01 04:00', f'2020-{m}-11 03:00'), {'freq': 'h'}) # args, kwargs for date_range.

# Path for downloaded ERA5 data.
FMT_PATH_ERA5_SFC = (
    PATHS['era5dir']
    / 'data/oper/an_sfc/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_sfc_{year}{month:02d}{day:02d}{hour:02d}00.{var}.nc'
)

# ERA5 base paths.
FMT_PATH_ERA5_ML = (
    PATHS['era5dir']
    / 'data/oper/an_ml/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.{var}.nc'
)

def era5_sfc_fmtp(var, year, month, day, hour):
    return util.format_path(FMT_PATH_ERA5_SFC, year=year, month=month, day=day, hour=hour, var=var)

def era5_ml_fmtp(var, year, month, day, hour):
    fmt = FMT_PATH_ERA5_ML
    return util.format_path(fmt, year=year, month=month, day=day, hour=hour, var=var)

