#!/bin/bash

#SBATCH --job-name=HurricaneMelissaReanalysis
#SBATCH --partition=seseml
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=ianpark3@illinois.edu
#SBATCH --output "/data/keeling/a/ianpark3/ATMS-Thesis/Dimensional-Reduction-in-Tropical-Cyclones/RunLogs/slurm-%A_driver.out"

now=$(date +"%T")
echo "Start time : $now"

module purge
module load python/3.14.3


ulimit -s -S unlimited

~/miniconda3/envs/ThesisTest/bin/python3.14 -u IndividualStormScript.py
now=$(date +"%T")
echo "End time : $now"

echo ""|mailx -s "Hurricane Analysis completed" ianpark3@illinois.edu