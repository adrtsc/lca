import zarr
from pathlib import Path
import sys
import yaml
from lca.ndt import top_level_features, top_level_features_zarr


# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

zarr_path = Path(settings['paths']['zarr_path'])

# load hdf5 file of site
filename = f'site_{site:04d}.zarr'
file = zarr.open(zarr_path.joinpath(filename), "r")

if len(file['intensity_images']['sdc-GFP']['level_00'].attrs['element_size_um']) == 2:
    # extract metadata and features
    top_level_features.main(file, settings)
elif len(file['intensity_images']['sdc-GFP']['level_00'].attrs['element_size_um']) == 3:
    # extract metadata and features
    top_level_features_zarr.main(file, filename,  settings)

