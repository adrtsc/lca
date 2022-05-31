import sys
from skimage import io
from pathlib import Path
from lca.ndt.segmentation import find_boundaries_2DT
import yaml
import h5py
import numpy as np
import re

# define the site this job should process
site = 1

# load settings
settings_path = Path(r'Y:\PhD\Code\Python\lca\scripts\settings\20211111_UAP56_settings.yml')
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])
channel = 'sdcRFP590-JF549'

with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:
    intensity_image = file[f'intensity_images/{channel}'][:, :, :, :]

from skimage.transform import pyramid_gaussian
import napari

intensity_image = np.moveaxis(intensity_image, source=0, destination=3)

pyramid = tuple(pyramid_gaussian(intensity_image,
                                 max_layer=3,
                                 downscale=2,
                                 preserve_range=True,
                                 channel_axis=-1))

new_pyramid = []
for layer in pyramid:
    layer = np.moveaxis(layer, source=3, destination=0)
    new_pyramid.append(layer)

new_pyramid = tuple(new_pyramid)

viewer = napari.Viewer()
viewer.add_image(new_pyramid, multiscale=True)



