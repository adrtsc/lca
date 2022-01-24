import h5py
from pathlib import Path
import sys
import yaml
from lca.ndt import top_level_features, top_level_features_3DT


# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])

# load hdf5 file of site
with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:

    if len(settings['spacing']) == 2:
        # extract metadata and features
        top_level_features.main(file, settings)
    elif len(settings['spacing']) == 3:
        # extract metadata and features
        top_level_features_3DT.main(file, settings)

