#! /bin/sh

# Submit a job array with index values between 0 and 5 (inclusive)
#SBATCH --array=1-4

#SBATCH -o /home/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /home/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#

### use (max) 2000 MB of memory per CPU
#SBATCH --mem-per-cpu=2000m

### use (max) 2000 MB of memory per CPU
#SBATCH --cpus-per-task=1

### time limit
#SBATCH --time=240

n="$SLURM_ARRAY_TASK_ID"

exec python -u hdf5_compression_illcorr_2DT_stack.py $n