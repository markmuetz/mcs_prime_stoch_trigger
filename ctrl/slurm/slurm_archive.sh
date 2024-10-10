#!/bin/bash
#SBATCH -p short-serial
#SBATCH -o slurm_archive/%J.out
#SBATCH -e slurm_archive/%J.err
#SBATCH -t 4:00:00

# Taken from: https://help.jasmin.ac.uk/article/4890-how-to-submit-a-job-to-slurm

# executable
cd /home/users/mmuetz/projects/mcs_prime_stoch_trigger/ctrl/remakefiles
remake archive archive.py
