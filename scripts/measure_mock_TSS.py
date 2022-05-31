import zarr
from pathlib import Path
import sys
import numpy as np
import random
import pandas as pd
import yaml
from lca.ndt.measure_new import measure_coordinates_3DT

# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

zarr_path = Path(settings['paths']['zarr_path'])
feature_path = Path(settings['paths']['feature_path'])

# load hdf5 file of site
filename = f'site_{site:04d}.zarr'
z = zarr.open(zarr_path.joinpath(filename), "r")

def random_points_mask(regionmask, n=500):
    mask_pixels = np.where(regionmask>0)
    rnd_centers = random.sample(range(1, len(mask_pixels[0])), n)

    random_blobs = pd.DataFrame()

    random_blobs['centroid-0'] = mask_pixels[0][rnd_centers]
    random_blobs['centroid-1'] = mask_pixels[1][rnd_centers]
    random_blobs['centroid-2'] = mask_pixels[2][rnd_centers]

    return random_blobs

img = z['intensity_images/sdc-RFP-605-52/level_00'][:]
nuclei = z['label_images/nuclei/level_00'][:]
speckles = z['label_images/nuclear_speckles/level_00'][:]

nucleoplasm = np.where(speckles == 1, 0, nuclei)

mock_TSS = []

for t in range(0, 60):
    print(t)
    random_point = random_points_mask(nucleoplasm[t, :, :, :], n=50000)
    random_point['timepoint'] = t

    labels = []

    for id, blob in random_point.iterrows():
        label = nucleoplasm[blob['timepoint'],
                            blob['centroid-0'],
                            blob['centroid-1'],
                            blob['centroid-2']]

        labels.append(label)

    random_point['label'] = labels

    mock_TSS.append(random_point)

mock_TSS = pd.concat(mock_TSS)
measured_blobs = measure_coordinates_3DT(mock_TSS, img, sigma=1.5)

# only keep the 5 percent highest intensity blobs (to remove the ones that land outside of cells or in nucleolus

filtered_blobs = measured_blobs.groupby(['timepoint', 'label'], as_index=False).apply(lambda g: g[g['mean_intensity'] > g['mean_intensity'].quantile(0.95)])

# sample one random blob per cell from the resulting population
filtered_blobs = filtered_blobs.groupby(['timepoint', 'label'], as_index=False).apply(lambda x: x.sample(1))

filtered_blobs.reset_index(inplace=True)
filtered_blobs.drop(['level_0', 'level_1', 'level_2'], axis=1, inplace=True)

filtered_blobs.to_csv(feature_path.joinpath(f'mock/20220224_mock_TSS_site_{site:04d}.csv'))