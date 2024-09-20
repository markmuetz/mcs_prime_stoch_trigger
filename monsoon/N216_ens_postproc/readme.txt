Scripts for extracting/combining selected vars (precip, TCWV, MCSP calling freq) into one .nc file.
And then transferring these to JASMIN using rsync.

Uses Monsoon pbs scheduler.
If I was smarter I would include these into the cylc workflow.
Although currently rsync tx requires eval ssh-agent and adding the JASMIN key.
