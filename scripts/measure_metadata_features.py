import h5py
from pathlib import Path
import sys
import yaml
from lca.ndt import top_level_features


# load settings
with open('settings/20210930_cluster_settings.yml', 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])

# define the site this job should process
site = int(sys.argv[1])

# load hdf5 file of site
file = h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a")

# extract metadata and features
top_level_features.main(file, settings)