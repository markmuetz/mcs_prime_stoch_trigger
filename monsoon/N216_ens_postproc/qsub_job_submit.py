import datetime as dt
from pathlib import Path
import subprocess as sp

from extract_combine_cubes import JOBS, gen_outpath

def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')

pbs_tpl = """
#!/bin/bash
#PBS -N {name}
#PBS -q shared
#PBS -o pbs_output/qsub_{ts}.{name}.out
#PBS -e pbs_output/qsub_{ts}.{name}.err
#PBS -l mem=8gb
#PBS -l walltime=02:00:00

module load scitools
which python
cd {cwd}
python extract_combine_cubes.py {case} {suite} {stash_code}
"""

def main():
    ts = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    for case, suite, stash_code in JOBS:
        outpath = gen_outpath(case, suite, stash_code)
        if outpath.exists():
            print(f'{outpath} already created')
            continue

        key = f'{case}_{suite}_{stash_code}'
        print(key)
        scriptpath = Path(f'pbs_scripts/script_{key}.sh')
        scriptpath.write_text(pbs_tpl.format(
            name=key,
            ts=ts,
            cwd=Path.cwd(),
            case=case,
            suite=suite,
            stash_code=stash_code,
        ))
        cmd = f'qsub {scriptpath}'
        print(cmd)
        print(sysrun(cmd).stdout)


if __name__ == '__main__':
    main()

