#! /bin/sh
#SBATCH --array=0-3
#SBATCH -o /home/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /home/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#SBATCH --mem-per-cpu=2000m
#SBATCH --cpus-per-task=1
#SBATCH --time=240

n="$SLURM_ARRAY_TASK_ID"

exec python -u hdf5_compression_illcorr_2DT_stack.py $n scripts\settings\20210930_settings.yml
