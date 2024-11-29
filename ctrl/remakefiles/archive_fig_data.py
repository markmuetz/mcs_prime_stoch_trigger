from remake import ArchiveV1, ArchiveV1Rule

archive = ArchiveV1(
    name='mcs_prime_stoch_trigger_fig_data',
    git_repo_url='http://github.com/markmuetz/mcs_prime_stoch_trigger',
    remakefile='N216_ens_analysis.py',
    author='Mark Muetzelfeldt',
    email='mark.muetzelfeldt@reading.ac.uk',
    archive_loc='/gws/nopw/j04/mcs_prime/mmuetz/tmp',
)

# Commented out tasks use the raw data, so archiving their inputs is huge (293 GB).
rules = [
    'PlotTotalPrecip',
    'PlotSpreadSkill',
    'PlotAutocorr',
    'PlotTCWV',
    'PlotMCSPCallingFreq',
]
for rule_name in rules:
    archive.add(
        ArchiveV1Rule(
            rule_name,
            inputs='all',
            # outputs='all',
        )
    )
