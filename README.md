# MCS:PRIME stochastic trigger analysis

* Author Mark Muetzelfeldt: mark.muetzelfeldt@reading.ac.uk
* project: MCS:PRIME


* Heart of this projct is in `ctrl/remakefiles`. 
* Env: pixi (`pixi.toml` + `pixi.lock`), conda-forge for esmpy/xesmf and mo_pack, with remake as an
  editable install of `../remake` (`~/projects/local/remake`). The old `upflo_env`/`upflo_remake3_env`
  conda envs are gone.
* On JASMIN:
  - `pixi install` (once)
  - `cd ctrl/remakefiles && pixi run remake run N216_ens_analysis.py -E slurm`
  - Always submit through `pixi run` (or from `pixi shell`): SLURM jobs inherit the submitting shell's
    PATH, and that is how they find this env's `remake`.
* Outputs go under `$DATADIR/$OUTPUT_TAG/` (set in `ctrl/remakefiles/proj_config.py`, currently
  `pixi_env`). Set `OUTPUT_TAG = None` for the canonical locations.
