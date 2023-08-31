import sys
import os
from pathlib import Path
import numpy as np
import yaml
import re


SLURM_COMMAND = """#! /bin/sh
#SBATCH --array=1-64
#SBATCH -o /home/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /home/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#SBATCH --mem-per-cpu=8000m
#SBATCH --cpus-per-task=1
#SBATCH --time=240

n="$SLURM_ARRAY_TASK_ID"

exec python zarr_compression_illcorr_per_site.py $n {0}
"""

# load settings

settings_path = Path(sys.argv[1])


with open("temp.sh", "w") as f:
    f.write(SLURM_COMMAND.format(settings_path))
os.system("sbatch temp.sh")
os.unlink("temp.sh")
