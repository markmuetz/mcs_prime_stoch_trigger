class ZenodoTarball(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs():
        inputs = {}
        for expt, var in expt_var:
            suite = EXPT_SIM[expt]
            # TODO:
            # inputs[f'input_{expt}_{var}'] = N216ExtractCombineVar.rule_outputs(expt, var)['output']
        return inputs

    @staticmethod
    def rule_outputs():
        outputs = {
            f'output': SIMDIR / f'N216ens.tar.gz'
        }
        return outputs


    def rule_run(self):
        outpath = self.outputs['output']
        inpaths = ' '.join(str(p) for p in self.inputs.values())
        cmd = f'tar czf {outpath} -C {SIMDIR} {inpaths}'
        print(cmd)
        sysrun(cmd)



