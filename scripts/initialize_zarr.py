import zarr
import numpy as np
from pathlib import Path
import re
import yaml
import sys

IMG_DIMS = [2160, 2560]
N_LEVELS = 4

# load settings
settings_path = Path(sys.argv[1])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# get all paths from settings
img_path = Path(settings['paths']['img_path'])
output_path = Path(settings['paths']['zarr_path'])
mag = settings['magnification']
file_extension = settings['file_extension']


img_files = img_path.glob(f'*.{file_extension}')
img_files = [fyle for fyle in img_files]


channel_names = np.unique(
    [re.search("C[0-9]{2}(?=.tif)",
               str(fyle)).group(0) for fyle in img_files])
n_sites = len(np.unique(
    [re.search("F[0-9]{3}(?=L)",
               str(fyle)).group(0) for fyle in img_files]))
n_slices = len(np.unique(
    [re.search("Z[0-9]{2,}(?=C)",
               str(fyle)).group(0) for fyle in img_files]))
n_tp = len(np.unique(
    [re.search("T[0-9]{4}(?=F)",
               str(fyle)).group(0) for fyle in img_files]))

chunk = (1, n_slices, *IMG_DIMS)


for site in range(1, n_sites+1):
    # Open the experiment zarr file
    if output_path.joinpath(f'site_{site:04}.zarr').exists():
        z = zarr.open(output_path.joinpath(f'site_{site:04}.zarr'), mode='a')
        print('opening in append mode')
    else:
        z = zarr.open(output_path.joinpath(f'site_{site:04}.zarr'), mode='w')
        print('making new file')
    for channel in channel_names:
        for level in range(0, N_LEVELS):
            if not hasattr(z, f'intensity_images/{channel}/level_{level}'):
                d = z.create_dataset(
                    f'intensity_images/{channel}/level_{level:02d}',
                    shape=[n_tp, n_slices,
                           IMG_DIMS[0]/2**level,
                           IMG_DIMS[1]/2**level],
                    chunks=chunk,
                    dtype='uint16')
        
                d.attrs["element_size_um"] = (1,
                                              6.5 / mag * 2 ** level,
                                              6.5 / mag * 2 ** level)