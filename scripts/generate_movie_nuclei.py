import zarr
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import napari

padding = 200
level = 2
channel = 'sdc-RFP-605-52'

settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20220224_settings.yml")
feature_path = Path(r"Z:\20220224_hiPSC_MS2\features\distance_measurements")

fv = pd.read_csv(feature_path.joinpath("20220224_feature_values_preprocessed_clustered_time.csv"))

with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

zarr_path = Path(settings['paths']['zarr_path'])

zarr_sample = zarr.open(zarr_path.joinpath('site_0001.zarr'), mode='r')
scaling = zarr_sample[f'intensity_images/sdc-GFP/level_{level:02d}'].attrs['element_size_um']

scale_corr = 0.8667


# take 5 sample tracks of each cluster

sample = fv.groupby(['cluster']).sample(3)['unique_track_id']

sample_df = fv[np.isin(fv['unique_track_id'], sample)]

# correct coordinates for the current level

sample_df['centroid-1'] = sample_df['centroid-1']*2**(3-level)
sample_df['centroid-2'] = sample_df['centroid-2']*2**(3-level)

sample_df['bbox-2'] = sample_df['bbox-2']*2**(3-level)
sample_df['bbox-5'] = sample_df['bbox-5']*2**(3-level)
sample_df['bbox-1'] = sample_df['bbox-1']*2**(3-level)
sample_df['bbox-4'] = sample_df['bbox-4']*2**(3-level)

# get maximal bounding box size across all tracks

min_coords = np.array(sample_df[['bbox-2', 'bbox-1']])

max_coords = np.array(sample_df[['bbox-5', 'bbox-4']])

bbox_max = max_coords-min_coords



sample_df['centroid-1'] = sample_df['centroid-1']/scale_corr + padding
sample_df['centroid-2'] = sample_df['centroid-2']/scale_corr + padding

sample_df['new_bbox-2'] = (sample_df['centroid-2'] - 2*bbox_max[:, 0].max()).astype(int)
sample_df['new_bbox-5'] = (sample_df['centroid-2'] + 2*bbox_max[:, 0].max()).astype(int)
sample_df['new_bbox-1'] = (sample_df['centroid-1'] - 2*bbox_max[:, 1].max()).astype(int)
sample_df['new_bbox-4'] = (sample_df['centroid-1'] + 2*bbox_max[:, 1].max()).astype(int)

out = []

for cl_lbl in sorted(sample_df['cluster'].unique()):

    cl_df = sample_df[sample_df['cluster'] == cl_lbl]
    print(f'cluster_{cl_lbl}')

    class_out = []

    for track_id in cl_df['unique_track_id'].unique():

        track = sample_df[sample_df['unique_track_id'] == track_id]

        print(f'track_{track_id}')

        out_track = []

        for timepoint in range(60):

            print(timepoint)
            track_tp = track[track['timepoint'] == timepoint]
            site = track_tp.site.unique()[0]
            z = zarr.open(zarr_path.joinpath(f'site_{site+1:04d}.zarr'), mode='r')

            x_low = int(track_tp['new_bbox-2'])
            x_high = int(track_tp['new_bbox-5'])
            y_low = int(track_tp['new_bbox-1'])
            y_high = int(track_tp['new_bbox-4'])

            #x_low = int(track_tp['bbox-2'])
            #x_high = int(track_tp['bbox-5'])
            #y_low = int(track_tp['bbox-1'])
            #y_high = int(track_tp['bbox-4'])

            label = int(track_tp['label'])

            lbl = z[f'label_images/nuclei/level_{level:02d}'][timepoint, :, :, :]
            lbl = np.pad(lbl, pad_width=[(0, 0), (padding, padding), (padding, padding)])
            lbl = lbl[:, y_low: y_high, x_low: x_high]
            lbl[lbl != label] = 0

            img = z[f'intensity_images/{channel}/level_{level:02d}'][timepoint, :, :, :]
            img = np.pad(img, pad_width=[(0, 0), (padding, padding),  (padding, padding)], constant_values=25)
            img = img[:, y_low: y_high, x_low: x_high]
            img[lbl != label] = 0


            out_track.append(img)

        class_out.append(np.stack(out_track))

    out.append(np.concatenate(class_out, axis=2))

img = np.concatenate(out, axis=3)

rfp = img
gfp = img

viewer = napari.Viewer()
viewer.add_image(img, scale=scaling, contrast_limits=(105, 350), colormap='magenta')

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.relplot(data=sample_df,
                 x='timepoint',
                 y='normalized_mean_intensity',
                 hue='unique_track_id',
                 linewidth=3, palette='tab20', kind='line',
                 col='cluster', col_wrap=2, aspect=2)
ax.set(ylabel='distance to ns border',
       xlabel='time (seconds)')

plt.show()