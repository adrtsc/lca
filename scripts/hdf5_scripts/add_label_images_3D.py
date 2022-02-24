import sys
from skimage import io
from pathlib import Path
from lca.ndt.segmentation import find_boundaries_2DT
import yaml
import h5py
import numpy as np
import re

# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])
temp_seg_path = Path(settings['paths']['temp_seg_path'])
file_extension = settings['file_extension']

img_files = temp_seg_path.glob(f'*_s{site}.tif')
img_files = [fyle for fyle in img_files]
objects = np.unique([re.search(f"[^\W]+(?=_s{site}.tif)",
                               str(fyle)).group(0) for fyle in img_files])

with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:
    reference = list(file['intensity_images'].keys())[0]
    element_size_um = file[f'intensity_images/{reference}'].attrs.get(
        'element_size_um')
    chunk = element_size_um = file[f'intensity_images/{reference}'].chunks

    for obj in objects:

        label_images = io.imread(temp_seg_path.joinpath(
            obj + f'_s{site}.tif'),
            plugin="tifffile")

        # Create a dataset in the file to add label images
        dataset = file['label_images'].create_dataset(obj,
                                                      np.shape(label_images),
                                                      h5py.h5t.STD_U16BE,
                                                      data=label_images,
                                                      compression='gzip', 
                                                      chunks=chunk,
                                                      shuffle=True,
                                                      fletcher32=True)

        dataset.attrs.create(name="element_size_um", data=element_size_um)

        # Create a dataset in the file to add boundaries
"""        boundaries = find_boundaries_2DT(label_images)
        dataset = file['boundary_images'].create_dataset(
            f'{obj}_boundaries',
            np.shape(label_images),
            h5py.h5t.STD_U16BE,
            data=boundaries,
            compression='gzip',
            chunks=chunk,
            shuffle=True,
            fletcher32=True)
"""
        dataset.attrs.create(name="element_size_um", data=element_size_um)
