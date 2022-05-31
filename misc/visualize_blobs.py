import zarr
from pathlib import Path
import sys
import numpy as np
import random
import pandas as pd
import yaml

site = 1

# load settings
settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20220224_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

zarr_path = Path(settings['paths']['zarr_path'])
feature_path = Path(settings['paths']['feature_path'])

# load hdf5 file of site
filename = f'site_{site:04d}.zarr'
z = zarr.open(zarr_path.joinpath(filename), "r")

RFP = z['intensity_images/sdc-RFP-605-52/level_00'][0, :, :, :]
GFP = z['intensity_images/sdc-GFP/level_00'][0, :, :, :]
scale = z['intensity_images/sdc-RFP-605-52/level_00'].attrs['element_size_um']

feature_path = Path(r"Z:\20220224_hiPSC_MS2\features")
fv = pd.read_csv(feature_path.joinpath("20220224_fv_preprocessed_step2.csv"))

fv = fv.loc[fv.site == 0]

import napari
viewer = napari.Viewer()

viewer.add_image(RFP)
viewer.add_image(GFP)
viewer.add_points(fv.loc[fv.mock == False][['timepoint',
                      'centroid-0_blobs',
                      'centroid-1_blobs',
                      'centroid-2_blobs']], face_color='transparent',
                  edge_color='black')

viewer.add_points(fv.loc[fv.mock == True][['timepoint',
                      'centroid-0_blobs',
                      'centroid-1_blobs',
                      'centroid-2_blobs']], edge_color='gray',
                  face_color='transparent')


viewer.add_points(mock_TSS[['timepoint',
                      'centroid-0',
                      'centroid-1',
                      'centroid-2']], edge_color='gray',
                  face_color='transparent')

viewer.add_points(filtered_blobs.reset_index()[['timepoint',
                      'centroid-0',
                      'centroid-1',
                      'centroid-2']], edge_color='gray',
                  face_color='transparent')