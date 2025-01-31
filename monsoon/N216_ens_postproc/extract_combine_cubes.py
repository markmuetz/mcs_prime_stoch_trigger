import os
import sys
from itertools import product
from pathlib import Path

import iris
from iris.util import equalise_attributes
import numpy as np

CASES = [
    # '20200101T0000Z',
    # '20200201T0000Z',
    # '20200301T0000Z',
    # '20200401T0000Z',
    # '20200501T0000Z',
    # '20200601T0000Z',
    '20200701T0000Z',
    # '20200801T0000Z',
    # '20200901T0000Z',
    # '20201001T0000Z',
    # '20201101T0000Z',
    # '20201201T0000Z',
]

SUITES = [
    # 'u-dg135',
    # 'u-di727',
    # 'u-di728',
    # 'u-dj618',
]

def cube_cell_method_is_empty(cube):
    return cube.cell_methods == tuple()

def cube_cell_method_is_not_empty(cube):
    return cube.cell_methods != tuple()

CONSTRAINTS = {
    'm01s05i216': (
        iris.AttributeConstraint(STASH='m01s05i216') &
        iris.Constraint(cube_func=cube_cell_method_is_empty)
    ),
    'm01s05i216.1h-mean': (
        iris.AttributeConstraint(STASH='m01s05i216') &
        iris.Constraint(cube_func=cube_cell_method_is_not_empty)
    ),
    'm01s05i993.1h-mean': iris.AttributeConstraint(STASH='m01s05i993'),
    'm01s30i461.1h-mean': iris.AttributeConstraint(STASH='m01s30i461'),
    'm01s16i202.500hPa': (
        iris.AttributeConstraint(STASH='m01s16i202') &
        iris.Constraint(pressure=500),  # constrained load to just get 500 hPa.
    ),
}

JOBS = list(product(CASES, SUITES, CONSTRAINTS))
# Add one case for u-dj618
JOBS.extend([
    ('20200701T0000Z', 'u-dj618', con)
    for con in CONSTRAINTS.keys()
])
# Filter out non-existent MCSP var for u-dj727 (ctrl).
JOBS = [
    (case, suite, constraint) for (case, suite, constraint) in JOBS
    if not (suite == 'u-di727' and constraint == 'm01s05i993.1h-mean')
]

def gen_outpath(case, suite, stash_code, stream='a'):
    return (Path(f'/projects/mcsprime/mamue/cylc-run/{suite}/share/cycle/{case}/engl/um') /
            Path(f'englaa_p{stream}.merged.{case}.{suite}.{stash_code}.nc'))


def run_job(case, suite, constraint_key, stream='a'):
    cwd = os.getcwd()
    os.chdir(f'/projects/mcsprime/mamue/cylc-run/{suite}/share/cycle/{case}/engl/um')

    stash_code = constraint_key
    if suite == 'u-di727' and stash_code == 'm01s05i993.1h-mean':
        print('There are no MCSP variables for the ctrl')
        print(f'{suite} {stash_code}')
        return
    constraint = CONSTRAINTS[stash_code]

    outpath = gen_outpath(case, suite, stash_code, stream=stream)
    if outpath.exists():
        print(outpath, 'already exists')
        raise Exception(f'{outpath} already exists')

    cubes_for_constraint = []
    for i in range(10):
        print(i)
        if stream == 'a':
            cube = iris.load_cube(f'em{i}/englaa_pa???.pp', constraint)
        elif stream == 'b':
            cube = iris.load_cube(f'em{i}/englaa_pb???.pp', constraint)
        if i == 0:
            coord = iris.coords.AuxCoord(points=np.array([0], dtype=np.int32), standard_name='realization', units='1')
            cube.add_aux_coord(coord)
        # if suite == 'u-di728' and stash_code in ['m01s05i993.1h-mean', 'm01s30i461.1h-mean']:
        # TODO:
        # Something weird going on with the EMs for this suite. EDIT: other suites as well.
        # Apply fix to all suites.
        # forecast_period and forecast_reference_time are messed up for EM5 (or perhaps all other EMs).
        # Investigate further.
        print('removing coords forecast_period and forecast_reference_time')
        cube.remove_coord('forecast_period')
        cube.remove_coord('forecast_reference_time')
        cube.attributes['notes'] = 'removing coords forecast_period and forecast_reference_time'

        cubes_for_constraint.append(cube)

    cubes_for_constraint = iris.cube.CubeList(cubes_for_constraint)
    equalise_attributes(cubes_for_constraint)
    merged_cube = cubes_for_constraint.merge_cube()
    iris.save(merged_cube, str(outpath))  # zlib=True is *really* slow.
    os.chdir(cwd)


if __name__ == '__main__':
    print(sys.argv)
    run_job(*sys.argv[1:])
