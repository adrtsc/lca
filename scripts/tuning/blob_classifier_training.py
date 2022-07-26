import zarr
import yaml
from pathlib import Path
import napari
import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk
import numpy as np

# pick random timepoints to train the classifier
timepoints = np.random.randint(0, 60, 5)

# load settings
settings_path = Path(r"/scripts/settings/20220218_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

site = 1

# define which level should be used
level = 0

channel = 'sdcRFP590-JF549'

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])
feature_path = Path(settings['paths']['feature_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

fyle = zarr_files[0]

z = zarr.open(fyle, mode='a')

img = [z['intensity_images'][channel][f'level_{level:02d}'][timepoint, :, :, :] for timepoint in timepoints]
img = np.stack(img)

scaling = z['intensity_images'][channel][f'level_{level:02d}'].attrs['element_size_um']

# load blobs
blob_files = feature_path.glob('*.csv')
blob_files = [blob_file for blob_file in blob_files if f'site_{site:04d}_blobs_' in str(blob_file)]


for blob_file in blob_files:

    blobs = pd.read_csv(blob_file)
    filter = blobs['timepoint'].isin(timepoints)
    blobs = blobs[filter]
    scaling_blobs = scaling.copy()
    scaling_blobs.insert(0, 1)

for idx, timepoint in enumerate(timepoints):
    blobs['timepoint'].loc[blobs['timepoint'] == timepoint] = idx


viewer = napari.Viewer()
viewer.add_image(img, scale = scaling)
viewer.add_points(blobs[['timepoint', 'centroid-0', 'centroid-1_scaled', 'centroid-2_scaled']], name=str(blob_file).replace('.csv', '').split('site_%04d_' % site)[1],
                      face_color='transparent',
                      edge_color='white',
                      size=blobs['size'],
                      visible=False,
                      scale=scaling_blobs)


import pickle

loaded_model = pickle.load(open(r"Z:\20220218_hiPSC_MS2\clf\new", 'rb'))

test = loaded_model.classify(blobs)
test = test[test.classification == 1]

viewer.add_points(test[['timepoint', 'centroid-0', 'centroid-1_scaled', 'centroid-2_scaled']], name=str(blob_file).replace('.csv', '').split('site_%04d_' % site)[1],
                      face_color='transparent',
                      edge_color='white',
                      size=test['size']/2**level,
                      visible=False,
                      scale=scaling_blobs)

from napari_blob_detection.measure_blobs import measure_coordinates


coordinates = viewer.layers.selection.active.data
sizes = viewer.layers.selection.active.size

measurements = measure_coordinates(coordinates, sizes, img)