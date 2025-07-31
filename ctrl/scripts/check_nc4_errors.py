import xarray as xr
from pathlib import Path

nc4_error_files = Path('nc4.err')
nc4_good_files = Path('nc4.good')

good = set(Path(p) for p in nc4_good_files.read_text().split('\n') if p)
bad = set(Path(p) for p in nc4_error_files.read_text().split('\n') if p)

paths = sorted(Path('/gws/nopw/j04/mcs_prime/mmuetz/data/GPM_IMERG_final/30min').glob('**/*.nc4'))
for path in paths:
    if path in good | bad:
        continue

    try:
        print(path)
        ds = xr.open_dataset(path)
        with nc4_good_files.open('a') as f:
            f.write(str(path) + '\n')
    except Exception as e:
        print('ERROR:', path, e)
        with nc4_error_files.open('a') as f:
            f.write(str(path) + '\n')
