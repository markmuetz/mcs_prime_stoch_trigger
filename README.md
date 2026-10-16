# MCS:PRIME stochastic trigger analysis

Analysis code for Muetzelfeldt et al., "An observationally constrained probabilistic trigger for
organized deep convection in an NWP ensemble" (submitted to QJRMS).

* Author: Mark Muetzelfeldt (mark.muetzelfeldt@reading.ac.uk)
* Project: MCS:PRIME (NERC grant NE/W005530/1)
* Licence: MIT (see `LICENSE`). To cite this code, see `CITATION.cff`.

The heart of this project is in `ctrl/remakefiles`: workflows for
[remake](https://github.com/markmuetz/remake) that process the N216 Unified Model ensemble
simulations and GPM IMERG V07 precipitation, and make the figures.

## Environment

* pixi (`pixi.toml` + `pixi.lock`). conda-forge is used for esmpy/xesmf and mo_pack.
* remake is an editable install of `../remake` (i.e. a clone of
  https://github.com/markmuetz/remake next to this repo, `~/projects/local/remake` on JASMIN).

## Running (on JASMIN)

* `pixi install` (once)
* `cd ctrl/remakefiles && pixi run remake run N216_ens_analysis.py -E slurm`
* Always submit through `pixi run` (or from `pixi shell`): SLURM jobs inherit the submitting shell's
  PATH, and that is how they find this env's `remake`.
* Outputs go under `$DATADIR/$OUTPUT_TAG/` (set in `ctrl/remakefiles/proj_config.py`, currently
  `pixi_env`). Set `OUTPUT_TAG = None` for the canonical locations.

The simulation output and IMERG data are not included. IMERG V07 Final half-hourly data can be
downloaded with `download_gpm_imerg.py` (needs a NASA Earthdata login in `~/.netrc`).

## Paper figures

Figure paths are relative to `$DATADIR/$OUTPUT_TAG/mcs_prime_figs/N216sims/prod/`.

| Figure | Remakefile and rule | Output |
|---|---|---|
| Fig. 1 | `N216_ens_analysis.py`: `plot_precip_snapshots` | `ensemble/20200201T0000Z/precip_snapshot/precip_snapshot.indian_ocean.t072.png` |
| Fig. 2 | `ASoP_analysis.py`: `plot_asop_n216regional` | `ASoP/all_cases/tropics/em1/asop.fractional_contrib.tropics.all_cases.3-hourly.em1.png` |
| Fig. 3 | `ASoP_analysis.py`: `plot_asop_n216regional` | `ASoP/all_cases/tropics/em1/asop.7x7_spat_temp_corr.tropics.all_cases.3-hourly.em1.pdf` |
| Fig. 4 | `N216_ens_analysis.py`: `plot_all_cases_spread_error` | `ensemble/all_cases/spread_error.all_cases.smooth=False-show_error_minus_spread=True-ens=red.cons.pdf` |
| Fig. S1 | `ASoP_analysis.py`: `plot_asop_n216regional` | `ASoP/all_cases/eq_warm_pool/em1/asop.fractional_contrib.eq_warm_pool.all_cases.3-hourly.em1.png` |
| Response to reviewers (round 2), Fig. 1 | `N216_ens_analysis.py`: `plot_all_cases_closest_member` | `ensemble/all_cases/closest_member_rmse.all_cases.ens=red.cons.pdf` |
