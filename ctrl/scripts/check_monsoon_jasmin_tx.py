from itertools import product
from pathlib import Path

simdir = Path('/gws/nopw/j04/mcs_prime/mmuetz/data/UM_sims')
suites = ['u-dg135', 'u-di727', 'u-di728']
cases = [f'2020{m:02d}01T0000Z' for m in range(1, 13)]
um_vars = [
    'm01s05i216',
    'm01s05i216.1h-mean',
    'm01s05i993.1h-mean',
    'm01s30i461.1h-mean',
]

merged_paths = []
for suite, case, var in product(suites, cases, um_vars):
    if suite == 'u-di727' and var == 'm01s05i993.1h-mean':
        continue
    path = simdir / f'{suite}/share/cycle/{case}/engl/um/englaa_pa.merged.{case}.{suite}.{var}.nc'
    merged_paths.append(path)

for path in merged_paths:
    if not path.exists():
        print('Not transferred yet:', path)
