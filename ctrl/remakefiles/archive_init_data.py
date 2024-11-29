from remake import ArchiveV1, ArchiveV1Rule

archive = ArchiveV1(
    name='mcs_prime_stoch_trigger_init_data',
    git_repo_url='http://github.com/markmuetz/mcs_prime_stoch_trigger',
    remakefile='N216_ens_analysis.py',
    author='Mark Muetzelfeldt',
    email='mark.muetzelfeldt@reading.ac.uk',
    archive_loc='/work/scratch-nopw2/mmuetz',
)

rules = [
    'GuassianFilterN216Imerg',
    'GuassianFilterExpt',
    'PlotMCSPCallingFreq',
    'PlotTCWV',
]
for rule_name in rules:
    archive.add(
        ArchiveV1Rule(
            rule_name,
            inputs='all',
        )
    )
