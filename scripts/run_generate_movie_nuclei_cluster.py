import os


SLURM_COMMAND = """#! /bin/sh
#SBATCH --array=0-{0}
#SBATCH -o /home/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /home/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#SBATCH --mem-per-cpu=6000m
#SBATCH --cpus-per-task=1
#SBATCH --time=480

n="$SLURM_ARRAY_TASK_ID"

exec python generate_movie_nuclei_cluster.py $n
"""

n_jobs = 6

with open("temp.sh", "w") as f:
    f.write(SLURM_COMMAND.format(n_jobs-1))
os.system("sbatch temp.sh")
os.unlink("temp.sh")