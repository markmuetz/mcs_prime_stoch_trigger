# Copying a file can fix these errors, perhaps by fixing striping issues on Lustre (according to ChatGPT).
from pathlib import Path
import shutil

import xarray as xr
nc4_error_files = Path('nc4.err')

bad = [Path(p) for p in nc4_error_files.read_text().split('\n') if p]

for path in bad:
    print('fixing', path)
    tmp = path.with_suffix('.tmp')

    # Copy to a temp file
    shutil.copy2(path, tmp)

    # Atomically replace the original file
    tmp.replace(path)

for path in bad:
    print('opening', path)
    ds = xr.open_dataset(path)

