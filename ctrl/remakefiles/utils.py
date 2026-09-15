from pathlib import Path
import shutil
import socket
import sys

import pandas as pd

import remake


def to_netcdf_tmp_then_copy(ds, outpath, encoding=None):
    """Helper to write output to a JASMIN scratch dir, then copy to desired location.

    Writing to this disk then copying is *MASSIVELY* faster.
    I think it is to do with writing NetCDF4 data to GWS is slow.

    Also, writes detailed metadata to netcdf.

    See: https://help.jasmin.ac.uk/article/176-storage
    Changed to nopw2 in July 23:
    /work/scratch-nopw[RO from 11July2023]
    """
    if encoding is None:
        encoding = {}
    # remake3 passes outputs['x'] as a path-like token, not a Path - wrap it
    # so the Path methods below (is_absolute/parts/parent) work.
    outpath = Path(outpath)
    tmpdir = Path('/work/scratch-nopw2/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    # Add some metadata to the netcdf file.
    # The direct caller is the rule function in the remakefile. Don't use
    # traceback.walk_stack(None): on Python 3.12 it starts several frames further up,
    # so it recorded remake's own remake_cmd.py instead of the remakefile.
    frame = sys._getframe(1)
    calling_file = frame.f_code.co_filename
    calling_function = frame.f_code.co_name
    # remake3: outpath is already the real output path (no tmp->actual map).
    output_path_actual = outpath

    nodename = socket.gethostname()
    remake_version = remake.__version__

    metadata_attrs = {
        'created by': f'{calling_file}: {calling_function}',
        'calling file source': Path(calling_file).read_text(),
        'project repository': 'https://github.com/markmuetz/mcs_prime_stoch_trigger',
        'remake version': remake_version,
        'remake repository': 'https://github.com/markmuetz/remake',
        # 'task': f'{calling_obj}',
        # 'task doc': f'{calling_obj_doc}',
        'created on': str(pd.Timestamp.now()),
        'nodename': nodename,
        'output path': str(output_path_actual),
        'contact': 'mark.muetzelfeldt@reading.ac.uk',
    }
    ds.attrs.update(metadata_attrs)

    ds.to_netcdf(tmppath, encoding=encoding)
    shutil.move(tmppath, outpath)
