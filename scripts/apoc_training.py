import zarr
import yaml
from pathlib import Path
import napari
import scipy.ndimage as ndi
import SimpleITK as sitk

# define the site this job should process
timepoint = 0

# load settings
settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20220414_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# define which level should be used for segmentation
level = 'level_00'

# define which channel should be used for segmentation
channel = 'C02'

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

fyle = zarr_files[1]

z = zarr.open(fyle, mode='a')

img = z['intensity_images'][channel][level][timepoint, :, :, :]

viewer = napari.Viewer()
viewer.add_image(img)
