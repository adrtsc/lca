import sys
import os
from pathlib import Path
import numpy as np
import yaml
import re


SLURM_COMMAND = """#! /bin/sh
#SBATCH --array=1
#SBATCH -o /home/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /home/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#SBATCH --mem-per-cpu=8000m
#SBATCH --cpus-per-task=1
#SBATCH --time=240

exec python initialize_zarr_well.py {1}
"""

# load settings

settings_path = Path(sys.argv[1])
n_wells = 40

with open("temp.sh", "w") as f:
    f.write(SLURM_COMMAND.format(n_wells, settings_path))
os.system("sbatch temp.sh")
os.unlink("temp.sh")
