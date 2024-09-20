class N216ExtractCombineVar(TaskRule):
    # No longer necessary because I do this on Monsoon.
    enabled = False
    """Extract and combine UM var at all times and for all EMs.
    """
    @staticmethod
    def rule_inputs(expt, var):
        suite = EXPT_SIM[expt]
        inputs = {
            f'pa_em{em_idx}_{h:03d}': SIMDIR / f'{suite}/share/cycle/20200701T0000Z/engl/um/em{em_idx}/englaa_pa{h:03d}.iris.nc'
            for em_idx in range(N_ENS_MEM)
            for h in range(0, 217, 24)
        }
        return inputs

    @staticmethod
    def rule_outputs(expt, var):
        suite = EXPT_SIM[expt]
        outputs = {
            f'output': SIMDIR / f'{suite}/processed/{expt}/engla_pa.{var}.nc'
        }
        return outputs


    var_matrix = {
        ('expt', 'var'): expt_var
    }

    def rule_run(self):
        def load_em_var(um_var, inputs, ens_idx):
            keep_coords = ['time', 'latitude', 'longitude']
            time_das = []

            for h in range(0, 217, 24):
                self.logger.debug(f'  Opening time {h}')

                em_path = inputs[f'pa_em{ens_idx}_{h:03d}']
                dsa = xr.open_dataset(em_path)
                da = dsa[um_var]
                # Why does time 0 have different names for all variables?
                if h == 0:
                    if um_var == 'precipitation_flux':
                        da = da.rename(time_0='time')
                    elif um_var in ['m01s05i993', 'm01s30i261']:
                        da = da.rename(time_1='time')
                else:
                    if um_var in ['m01s05i993', 'm01s30i261']:
                        da = da.rename(time_0='time')
                # Drop all variables that are not needed. This means concat will work.
                # (This drops all other coords with _0 suffix.)
                coord_names = [c.name for c in list(da.coords.values())]
                drop_coords = sorted(set(coord_names) - set(keep_coords))
                da = da.drop_vars(drop_coords)

                time_das.append(da)

            pflux = xr.concat(time_das, dim='time')
            return pflux.load()

        em_pfluxes = []
        if self.var == 'precip':
            # What's the difference between the two fluxes?
            # dsa.precipitation_flux has no time mean (i.e. it's instantaneous)
            # dsa.precipitation_flux_0 has 1-hr time mean.
            # I think it's better to use instantaneous to e.g. compare with IMERG.
            um_var = 'precipitation_flux'
        elif self.var == 'mcsp_calling_freq':
            # This uses a 1-h time mean.
            um_var = 'm01s05i993'
        elif self.var == 'tcwv':
            # This uses a 1-h time mean.
            um_var = 'm01s30i261'

        for ens_idx in range(N_ENS_MEM):
            self.logger.info(f'Loading EM {ens_idx}')
            em_pfluxes.append(load_em_var(um_var, self.inputs, ens_idx))

        da = xr.concat(em_pfluxes, dim=pd.Index(range(N_ENS_MEM), name='ens_mem'))
        da.attrs['UM simulation'] = EXPT_SIM[self.expt]
        da.attrs['MCS:PRIME expt'] = self.expt
        if self.var == 'mcsp_calling_freq':
            da.attrs['UM name'] = um_var
            da.rename(self.var)

        cu.to_netcdf_tmp_then_copy(da, self.outputs['output'])



