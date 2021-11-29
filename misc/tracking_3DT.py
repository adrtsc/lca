import h5py
from pathlib import Path
from lca.ndt.measure import measure_morphology_3DT
from lca.ndt.measure import measure_intensity_3DT
from lca.ndt.measure import measure_tracks_2DT
import napari

hdf5_path = Path(r"Z:\20211111_hiPSC_MS2\GFP_5_RFP_15_4s\short\hdf5\site_0001.hdf5")

with h5py.File(hdf5_path, "r") as file:
    labels = file['label_images/nuclei'][0:2, :, :, :]
    img = file['intensity_images/sdcGFP'][0:2, :, :, :]

df = measure_morphology_3DT(labels.astype('uint16'), [1 , 1, 6.15])
df = measure_intensity_3DT(labels.astype('uint16'), img, [1 , 1, 6.15])

test = measure_tracks_2DT(df, max_distance=100, time_window=3,
                          max_split_distance=150,
                          max_gap_closing_distance=200)


tracks = test[['track_id', 'timepoint', 'centroid-3', 'centroid-0', 'centroid-1']]
viewer = napari.Viewer()

viewer.add_labels(labels.astype('uint16'), scale=(6.15, 1, 1))
viewer.add_tracks(tracks)
viewer.add_points(df[['Centroid_z', 'Centroid_y', 'Centroid_x']])



features = ('label', 'area', 'bbox', 'bbox_area', 'centroid',
                'convex_area', 'eccentricity', 'equivalent_diameter',
                'euler_number', 'extent', 'major_axis_length',
                'minor_axis_length', 'moments', 'moments_central',
                'moments_hu', 'moments_normalized', 'orientation',
                'perimeter', 'solidity')

import numpy as np
from skimage.measure import regionprops_table
import pandas as pd

hdf5_path = Path(r"Z:\20211109_clone_9\hdf5\site_0001.hdf5")

with h5py.File(hdf5_path, "r") as file:
    labels = file['label_images/nuclei'][0, :, :, :]
    img = file['intensity_images/sdcGFP'][0:2, :, :, :]


test = pd.DataFrame(regionprops_table(labels[0,:,:,:], properties=features))