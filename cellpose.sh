#! /bin/sh

# Submit a job array with index values between 0 and 5 (inclusive)
#SBATCH --array=0-49

#SBATCH -o /data/homes/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /data/homes/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#

### use (max) 2000 MB of memory per CPU
#SBATCH --mem-per-cpu=20000m

### set max runtime
#SBATCH --time=240

### use (max) 2000 MB of memory per CPU
#SBATCH --cpus-per-task=1

n="$SLURM_ARRAY_TASK_ID"

exec python -u cellpose_2D_nuclei.py $n