from pathlib import Path
import h5py
import pandas as pd
import yaml
from skimage.draw import disk as draw_disk
from skimage.morphology import erosion, disk
import numpy as np
from lca.ndt.segmentation import find_boundaries_2DT
import sys

# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])
feature_path = Path(settings['paths']['feature_path'])

blobs = pd.read_csv(feature_path.joinpath('site_%04d_blobs_cytoplasm_sdcGFP.csv' % site))


with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:
    lbl_img = file['label_images/cytoplasm'][:]

new_lbl = np.zeros(np.shape(lbl_img))

for idx, lbl in enumerate(list(lbl_img)):

    c_blobs = blobs.loc[blobs['timepoint'] == idx]

    for index, blob in c_blobs.iterrows():

        rr, cc = draw_disk(tuple(blob[['centroid-0', 'centroid-1']]), blob['size'],
                           shape=np.shape(lbl))

        lbl[rr, cc] = 0

    new_lbl[idx, :, :] = erosion(lbl, disk(3))

    print(idx)


object = "cytoplasm_eroded"

with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:

    if 'cytoplasm_eroded' in file['label_images'].keys():
        del file['label_images/cytoplasm_eroded']

    element_size_um = file['intensity_images/%s' % list(file['intensity_images'].keys())[0]].attrs.get('element_size_um')
    chunk = element_size_um = file['intensity_images/%s' % list(file['intensity_images'].keys())[0]].chunks


    # Create a dataset in the file to add label images
    dataset = file['label_images'].create_dataset(object, np.shape(new_lbl), h5py.h5t.STD_U16BE, data=new_lbl,
                                 compression='gzip', chunks=chunk, shuffle=True, fletcher32=True)

    dataset.attrs.create(name="element_size_um", data=element_size_um)

    # Create a dataset in the file to add boundaries
    boundaries = find_boundaries_2DT(new_lbl)
    dataset = file['boundary_images'].create_dataset('%s_boundaries' % object, np.shape(new_lbl), h5py.h5t.STD_U16BE,
                                                     data=boundaries, compression='gzip', chunks=chunk,
                                                     shuffle=True, fletcher32=True)

    dataset.attrs.create(name="element_size_um", data=element_size_um)

