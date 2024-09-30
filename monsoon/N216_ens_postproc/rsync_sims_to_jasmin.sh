
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    # Reason: if you source it, then the ssh-agent will live beyond the life of the script.
    # And rerun will not require re-entering password.
    echo "Hey, you should source this script, not execute it!"
    exit 1
fi

if ps -p $SSH_AGENT_PID > /dev/null 2>&1
then
    echo "ssh-agent is already running"
    # Do something knowing the pid exists, i.e. the process with $PID is running
else
    echo "Initialize ssh-agent"
    eval $(ssh-agent -s)
    ssh-add ~/.ssh/id_rsa_jasmin
fi

cd /projects/mcsprime/mamue/cylc-run
rsync -Rav --progress ./{u-dg135,u-di727,u-di728}/share/cycle/2020??01T0000Z/engl/um/englaa_pa.merged.2020??01T0000Z.*.nc mmuetz@xfer3.jasmin.ac.uk:/gws/nopw/j04/mcs_prime/mmuetz/data/UM_sims/
cd -
