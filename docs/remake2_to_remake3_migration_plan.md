# remake2 → remake3 migration plan: `ctrl/remakefiles/`

Migration of the project's production remakefiles from remake2 (0.7.0) to
remake3, on branch `remake3_migration`. The purpose is a **side-by-side
equivalence test**: remake3 must reproduce the remake2 outputs. So the
migrated files are redirected to a parallel output tree and run to
completion, then diffed against the existing remake2 outputs (this is *not*
the usual "adopt existing outputs, expect ~0 runnable" migration — see
Acceptance test).

Follows the workflow in the remake3 skill's `remake2_to_remake3.md`:
translate by hand, wire `depends_on` explicitly (against the ground-truth
DAG in `docs/remake2_rule_dags.txt`), declare module globals via `uses=`.

## Done already on this branch

- Deleted `ctrl/remakefiles/dev/` (scratch/experimental Style-B files).
- Deleted `archive_init_data.py` and `archive_fig_data.py` — they use
  `ArchiveV1`/`ArchiveV1Rule`/`remake archive`, all **removed in remake3**
  with no equivalent.
- Added `ctrl/remakefiles/dump_dag.py` (remake2 DAG-interrogation helper,
  see Ground-truth DAG) and captured its output in
  `docs/remake2_rule_dags.txt`.

## Environment notes

- **remake3 CLI** is not on `PATH`; use `~/projects/remake3/.venv/bin/remake`.
- **remake2** (for the DAG dump / cross-checking) is in the `upflo_env`
  conda env: `~/miniforge3/envs/upflo_env/bin/python` (remake 0.7.0).
- **Drop the `mcs_prime` package dependency.** `download_era5.py` imports
  `mcs_prime.mcs_prime_config_util as cu`; that package fails to even
  import here (its config module stats a stale GWS path —
  `OSError: Stale file handle: /gws/nopw/j04/mcs_prime/...`). Replace `cu`
  usage with the local `proj_config` (`conf`). `cu.PATHS['datadir']` →
  `conf.PATHS['datadir']`. Verify any other `cu.*` names have a
  `proj_config` equivalent; add them to `proj_config.py` if missing. No
  migrated file should import `mcs_prime`.

## Ground-truth rule DAG (from remake2)

`dump_dag.py` loads a remakefile via remake2's `load_remake(..., finalize=
False)` and prints `rmk.rule_dg` (the rule-level `networkx.DiGraph`) nodes,
edges and topological order. Use this as the authoritative wiring target
when adding `depends_on` — the migrated remake3 DAG must match it. Full
output is in `docs/remake2_rule_dags.txt`; summary:

**`N216_ens_analysis.py` — 19 registered rules, 18 edges** (the file
defines 23 classes; **4 are `enabled = False`**: `CalcAutocorrImerg`,
`CalcAutocorrExpt`, `PlotAutocorr`, `PlotTCWV` — remake2 skips them, and
they do not appear in the DAG).

```
RegridImergToN216  -> CalcTotalPrecip, GuassianFilterN216Imerg, PlotPrecipSnapshots
RegridERA5ToN216   -> CalcTCWV, Calc_geopot_eRMSE
GuassianFilterExpt -> Calc_eRMSE, Calc_dRMSE
GuassianFilterN216Imerg -> Calc_eRMSE
CalcTotalPrecip    -> PlotTotalPrecip
Calc_eRMSE, Calc_dRMSE             -> PlotSpreadError, PlotAllCasesSpreadError
Calc_geopot_eRMSE, Calc_geopot_dRMSE -> PlotGeopotSpreadError, PlotAllCasesGeopotSpreadError
CalcMCSPCallingFreqData -> PlotMCSPCallingFreq
```
Isolated (no edges): `FirstLookPlotERA5_500hPa_geopotential`.

**`ASoP_analysis.py` — 3 rules, 1 edge:** `ASoPN216regional ->
PlotASoPN216regional`; `PlotRegions` isolated.

**`download_gpm_imerg.py` — 1 rule, 0 edges.**
**`download_era5.py` — 1 rule, 0 edges** (couldn't auto-dump due to the
`mcs_prime` import failure above; trivially a single standalone rule).

## Output redirection for the equivalence test

The equivalence test needs remake3 to write to a **separate tree** so the
remake2 outputs survive for diffing. Outputs derive from two keys in
`proj_config.py`:

```python
'outdir': .../mcs_prime_output            # NetCDF intermediate + final data
'figdir': .../mcs_prime_figs/N216sims/prod # figures
```

Redirect both to parallel `*_remake3` locations (single edit point), e.g.
`mcs_prime_output_remake3` and `.../N216sims/prod_remake3`. `datadir`
(raw inputs: UM sims, IMERG, ERA5) stays shared — remake3 reads the same
inputs. (`download_*` rules write into `datadir`; since those inputs
already exist, keep them pointed at the shared tree and simply don't re-run
the download rules during the equivalence test.)

## Migration order

Shared modules first (everything imports them), then production remakefiles
smallest → largest so the translation pattern is validated before the
1685-line, densely-wired file:

1. **`proj_config.py`** — add any `cu.*` names the de-`mcs_prime`'d files
   need; add the `*_remake3` output redirection. Confirm
   `remake.util.format_path` still exists in remake3.
2. **`utils.py`** — fix `to_netcdf_tmp_then_copy`: wrap `outpath =
   Path(outpath)` at the top (remake3 `outputs['x']` are path-like tokens,
   not `Path`; the function calls `.is_absolute()`/`.parts`). One fix
   covers all 15 call sites. Confirm `remake.__version__` and
   `remake.util.tmp_to_actual_path` exist in remake3.
3. **`download_gpm_imerg.py`** — 1 rule, 0 deps. First real translation;
   shake out `@rule` + `uses=` + SLURM-key + Path-wrap on a self-contained
   file.
4. **`download_era5.py`** — 1 rule; also drops `mcs_prime` → `proj_config`.
5. **`ASoP_analysis.py`** — 3 rules; first `depends_on` edge + shared
   module-level helper (`open_precip`). Validates the pattern at mid scale.
6. **`N216_ens_analysis.py`** — last. 19 rules / 18 edges + 4
   translate-but-don't-register disabled rules; applies every pattern at
   scale.

## Cross-cutting changes (every file)

- **Class → function.** `class R(Rule):` + `@staticmethod rule_run` →
  module-level `@rule(inputs=, outputs=, matrix=, uses=, depends_on=)` on a
  plain function. `rule_matrix`→`matrix=`; `rule_inputs`/`rule_outputs`
  (dict or `@staticmethod` callable) → `inputs=`/`outputs=` (same forms).
- **Explicit registration as the last line.** Use the **explicit list**
  form, not `rmk.rules_from_current_module()`, so the 4 disabled
  `N216_ens_analysis.py` rules can be translated (kept for `ScopeWarning`
  checking and so the logic isn't lost) but **omitted** from
  `rmk.add_rules([...])`. The single-rule download files can use
  `rmk.rules_from_current_module()`.
- **Explicit `depends_on`,** wired to match `docs/remake2_rule_dags.txt`.
  Cross-rule refs like `RegridImergToN216.rule_outputs(case, m)['output']`
  → `regrid_imerg_to_n216.outputs(...)` + a `depends_on=[regrid_imerg_to_n216]`
  entry. The `rule_inputs = CalcTotalPrecip.rule_outputs` shorthand (3
  sites: lines 431, 1120, 1215 — note 1120 is `PlotTCWV`, a disabled rule)
  → `inputs=calc_total_precip.outputs`.
- **`uses=` for module globals** each rule body references: `rmse`,
  `settings`/`Settings` (dataclass — supported in `uses=`), the `plot_*`
  helpers, `REGIONS`, `open_precip`, `client`, `req_var_names`,
  `GpmDatetime`, `gen_dates_urls_filenames`, `get_from_gpm`, templates,
  `DATES_KWARGS`. Imported modules (`np`, `xr`, `conf`, `utils`,
  `ASoPlite`, `scipy.stats`) are exempt. `uses` tracking is one level deep.
- **SLURM config translation.** `queue: 'short-serial'/'short-serial-4hr'`
  → `partition='standard', qos='standard'`; `max_runtime` → `time`. `mem`
  int (`64000`) → confirm remake3 accepts int MB or use a string (`'64G'`;
  remake3 default is `'4G'`). `download_*` still use old keys;
  `N216_ens_analysis.py` and `ASoP_analysis.py` already use `partition`/`qos`.
- **Drop `content_checks=False`** from `Remake(config=...)` — not a
  recognised remake3 arg (remake3 does no content/mtime hashing). Output
  adoption is governed by the top-level `check_outputs=` kwarg
  (default `'fallback'`).
- **Path-wrap output tokens** for `.parent`/`.touch()`/`.write_text()`/
  `/`-join: `N216_ens_analysis.py:253` (`outputs['dummy'].parent`), `:1391`
  (`.touch()`), `download_gpm_imerg.py:146,165`, plus the `utils` fix.

## Difficulties / risks (ranked)

1. **`N216_ens_analysis.py` scale + DAG density** — the bulk of the work;
   wire all 18 edges against the dump, handle the 4 disabled rules, and the
   dataclass/Path-wrap details.
2. **Equivalence diffing is the real acceptance bar,** not import-clean.
   Need a NetCDF comparison (e.g. `xr.testing.assert_allclose` / `nccmp`)
   tolerant of metadata that legitimately differs — `to_netcdf_tmp_then_copy`
   stamps `remake version`, `created on`, `nodename`, `calling file source`
   into `ds.attrs`, so an exact byte/attr diff will always differ. Compare
   **data variables**, ignore those attrs. Figures: compare by eye or
   rendered-pixel hash, not file bytes.
3. **`mcs_prime` removal** — must find `proj_config` equivalents for every
   `cu.*` reference, or the file won't import.
4. **`utils.to_netcdf_tmp_then_copy`** token-vs-Path + `remake.__version__`/
   `tmp_to_actual_path` availability in remake3.
5. **`content_checks=False` semantics** — confirm dropping it is inert
   (it should be; remake3 never content-hashes).
6. **SLURM `mem` int format** — verify int MB is accepted.

## Acceptance test (equivalence, per file)

Because outputs are redirected to an empty `*_remake3` tree, the plan
**should show all tasks runnable** (the opposite of a normal migration).

```
~/projects/remake3/.venv/bin/remake run <file> -n          # imports clean; DAG matches the dump
~/projects/remake3/.venv/bin/remake run <file> -Q '<one task>'  # one slice
# diff the redirected remake3 output against the remake2 output for that task
#   NetCDF: compare data vars, ignore remake/created-on/nodename attrs
#   figures: visual / pixel-hash
~/projects/remake3/.venv/bin/remake run <file>             # full run, then full-tree diff
```

Keep each remake2 file until its equivalence diff passes.

## Out of scope (this branch)

`stoch_trig_remakefiles/*` (Style-B `TaskRule`, 11-rule
`N216_sims_process.py` + `convert_pp_to_nc.py` + `download_gpm_imerg.py`,
plus its own `dev/`). Rule names overlap heavily with
`N216_ens_analysis.py` → almost certainly a superseded older pipeline.
Leave on remake2 unless confirmed still live; it would need the heavier
`self.`→explicit-params rewrite and also depends on `mcs_prime`.
