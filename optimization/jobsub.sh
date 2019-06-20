#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=haswell
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
./run.sh

