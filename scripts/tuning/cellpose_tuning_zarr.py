import zarr
import yaml
from pathlib import Path
import napari
import scipy.ndimage as ndi
import SimpleITK as sitk

# define the site this job should process
timepoint = 0

# load settings
settings_path = Path(r"/scripts/settings/20220317_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# define which level should be used for segmentation
level = 'level_03'

# define which channel should be used for segmentation
channel = settings['cellpose']['nuclei']['channel']

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

fyle = zarr_files[0]

z = zarr.open(fyle, mode='a')

img = z['intensity_images'][channel][level][timepoint, :, 0:100, 0:100]

#img = ndi.median_filter(img, 3)
filter_sigma=1
image = sitk.GetImageFromArray(img)
image.SetSpacing((1.0, 1.0, 1.25))
gaussianfilter = sitk.SmoothingRecursiveGaussianImageFilter()
gaussianfilter.SetSigma(filter_sigma)
img = sitk.GetArrayFromImage(gaussianfilter.Execute(image))

filter = sitk.MedianImageFilter()
filter.SetRadius(3)

viewer = napari.Viewer()
viewer.add_image(img)
