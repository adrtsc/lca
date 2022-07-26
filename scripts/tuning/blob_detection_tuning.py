import zarr
import yaml
from pathlib import Path
import napari
import scipy.ndimage as ndi
import SimpleITK as sitk
import numpy as np

# define the site this job should process
timepoint = 0

# load settings
settings_path = Path(r"/scripts/settings/20220218_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# define which level should be used for segmentation
level = 'level_00'

# define which channel should be used for segmentation
channel = settings['cellpose']['nuclei']['channel']

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

fyle = zarr_files[0]

z = zarr.open(fyle, mode='a')

img = z['intensity_images'][channel][level][0:3, :, :, :]
spacing = z['intensity_images'][channel][level].attrs['element_size_um']

img = np.expand_dims(img, 0)


viewer = napari.Viewer()
viewer.add_image(img, scale = spacing)
