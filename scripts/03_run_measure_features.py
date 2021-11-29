import sys
import os
from pathlib import Path
import numpy as np
import yaml
import re


SLURM_COMMAND = """#! /bin/sh
#SBATCH --array=1-{0}
#SBATCH -o /home/atschan/PhD/slurm_reports/slurm-%A_%a.out
#SBATCH -e /home/atschan/PhD/slurm_reports/slurmerror-%A_%a.out
#SBATCH --mem-per-cpu=7500m
#SBATCH --cpus-per-task=8
#SBATCH --time=240

n="$SLURM_ARRAY_TASK_ID"

exec python measure_features.py $n {1}
"""

# load settings

settings_path = Path(sys.argv[1])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

file_extension = settings['file_extension']
img_path = Path(settings['paths']['img_path'])
img_files = img_path.glob('*.%s' % file_extension)
img_files = [fyle for fyle in img_files]

# check if image files contain multiple sites
if any([bool(re.search('(?<=_s)[0-9]{1,}', str(fyle))) for fyle in img_files]):
    n_sites = len(np.unique([int(re.search("(?<=_s)[0-9]{1,}", str(fyle)).group(0)) for fyle in img_files]))
else:
    n_sites = 1

with open("temp.sh", "w") as f:
    f.write(SLURM_COMMAND.format(n_sites, settings_path))
os.system("sbatch temp.sh")
os.unlink("temp.sh")
