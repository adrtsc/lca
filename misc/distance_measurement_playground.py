import napari
from lca.nd.measure import measure_blobs_2D
from lca.ndt.measure import measure_blobs_3DT
import h5py
import yaml
from pathlib import Path

settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20211111_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

channel = 'sdcRFP590-JF549'
site = 1
hdf5_path = Path(settings['paths']['hdf5_path'])

n_tp = 30

with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:
    intensity_image = file[f'intensity_images/{channel}'][0:n_tp, :, :, :]
    label_image = file['label_images/nuclei/'][0:n_tp,:,:,:]
    speckles = file['label_images/nuclear_speckles/'][0:n_tp,:,:,:]
    speckles_intensity = file['intensity_images/sdcGFP'][0:n_tp, :, :, :]


viewer = napari.Viewer()
viewer.add_image(intensity_image, scale = [6.15, 1, 1])
viewer.add_labels(label_image.astype('uint16'), scale = [6.15, 1, 1])
viewer.add_labels(speckles.astype('uint16'), scale = [6.15, 1, 1])
viewer.add_image(speckles_intensity, scale = [6.15, 1, 1])

test = measure_blobs_3DT(intensity_image,
                         label_image,
                         min_sigma=2,
                         max_sigma=2,
                         num_sigma=1,
                         threshold=0.0005)


viewer.add_points(test[['timepoint',
                        'centroid-0',
                        'centroid-1',
                        'centroid-2']],
                  scale = [1, 6.15, 1, 1])


import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def measure_distance_points_mask_3D(df, mask, anisotropy):

    df['centroid-0'] = df['centroid-0']*anisotropy

    seg_coords = pd.DataFrame()
    seg_coords['z'], seg_coords['y'], seg_coords['x'] = np.where(mask == 1)
    seg_coords['z'] = np.round(seg_coords['z']*anisotropy)

    tree = KDTree(seg_coords)
    dist, points = tree.query(df[['centroid-0', 'centroid-1', 'centroid-2']], 1)
    df['distance_object'] = dist

    # same for distance to background (or how deep in a speckle they are)
    bg_coords = pd.DataFrame()
    bg_coords['z'], bg_coords['y'], bg_coords['x'] = np.where(mask == 0)
    bg_coords['z'] = np.round(bg_coords['z'] * anisotropy)

    tree_bg = KDTree(bg_coords)
    dist, points = tree_bg.query(df[['centroid-0', 'centroid-1', 'centroid-2']], 1)
    df['distance_bg'] = dist

    return df


def measure_distance_points_mask_3DT(df, mask, anisotropy):

    timepoints = np.unique(df['timepoint'])
    output = []

    for timepoint in timepoints:

        current_mask = mask[timepoint, :, :, :]
        current_df = df.loc[df['timepoint'] == timepoint]

        current_df['centroid-0'] = np.round(current_df['centroid-0'] * anisotropy)

        seg_coords = pd.DataFrame()
        seg_coords['z'], seg_coords['y'], seg_coords['x'] = np.where(current_mask == 1)
        seg_coords['z'] = np.round(seg_coords['z']*anisotropy)

        tree = KDTree(seg_coords)
        dist, points = tree.query(
            current_df[['centroid-0', 'centroid-1', 'centroid-2']], 1)
        current_df['distance_object'] = dist

        # same for distance to background (or how deep in a speckle they are)
        bg_coords = pd.DataFrame()
        bg_coords['z'], bg_coords['y'], bg_coords['x'] = np.where(current_mask == 0)
        bg_coords['z'] = np.round(bg_coords['z'] * anisotropy)

        tree_bg = KDTree(bg_coords)
        dist, points = tree_bg.query(
            current_df[['centroid-0', 'centroid-1', 'centroid-2']], 1)
        current_df['distance_bg'] = dist
        output.append(current_df)

    return pd.concat(output)


new_df = measure_distance_points_mask_3DT(test, speckles, 7.7)

feature_path = Path(r"Z:\20211111_hiPSC_MS2\GFP_5_RFP_15_4s\short\features\site_0001_nuclei_feature_values.csv")
fv = pd.read_csv(feature_path)

# add track id to blobs
track_ids = []
for idx, blob in new_df.iterrows():
    timepoint = blob.timepoint
    label = blob.label
    track_id = int(fv['track_id'].loc[(fv.timepoint == timepoint) &
                                      (fv.label == label)])

    track_ids.append(track_id)

new_df['track_id'] = track_ids

import seaborn as sns
import matplotlib.pyplot as plt

feature_path = Path(r'Z:\20211111_hiPSC_MS2\GFP_5_RFP_15_4s\short\features')

new_df = pd.read_csv(feature_path.joinpath('measured_distance.csv'))

sns.set_theme()

to_plot = new_df
to_plot['distance_speckles_um'] = to_plot['distance_object']*0.065

ax = sns.scatterplot(data=to_plot, x='mean_intensity', y='distance_speckles_um')
ax.set(ylabel="distance from nuclear speckle border ($\mu m)$",
       xlabel='mean transcriptional start site intensity')
plt.show()

aggregated_df = new_df.groupby(['track_id', 'timepoint']).mean().reset_index()

aggregated_df['distance_speckles_um'] = aggregated_df['distance_object']*0.065
aggregated_df['time_seconds'] = aggregated_df['timepoint']*6

sns.scatterplot(data=aggregated_df, x='timepoint', y='distance_object', hue='track_id', palette='tab20', size='mean_intensity')
plt.show()


ax = sns.lmplot(data=aggregated_df, x='mean_intensity',
                y='distance_speckles_um', col='track_id', col_wrap=2,
                hue='track_id', palette='tab10')
ax.set(ylabel="distance from nuclear speckle border ($\mu m)$",
       xlabel='normalized mean transcriptional start site intensity')
plt.show()

cell_2 = aggregated_df.loc[aggregated_df.track_id==6]

sns.scatterplot(data=cell_2, x='time_seconds', y='distance_object', hue='mean_intensity', palette='viridis')
sns.lineplot(data=cell_2, x='time_seconds', y='distance_object', alpha=0.1, color='black')
plt.show()

# normalize mean intensity per cell

def max_norm(group):
    group['mean_intensity'] = (group['mean_intensity'] - group['mean_intensity'].min()) / (group['mean_intensity'].max() - group['mean_intensity'].min())

    return group

aggregated_df = aggregated_df.groupby('track_id').apply(lambda group: max_norm(group))

# drop track id 1 and 11

aggregated_df = aggregated_df[~aggregated_df.track_id.isin([1, 11])]

ax = sns.relplot(data=aggregated_df, x='time_seconds',
                 y='distance_speckles_um',
                 hue='mean_intensity',
                 s=50,
                 aspect=1,
                 col='track_id', col_wrap=2, kind='scatter', palette='viridis',
                 legend=False)
ax.set(ylabel="distance from nuclear speckle border ($\mu m)$",
       xlabel='time (s)')
sm = plt.cm.ScalarMappable(cmap="viridis")
plt.colorbar(sm)
plt.show()

summary_measurements = pd.DataFrame()
summary_measurements['fano_factor'] = aggregated_df.groupby('track_id').apply(lambda x: np.var(x['mean_intensity']/np.mean(x['mean_intensity'])))
summary_measurements['mean_distance_ns'] = aggregated_df.groupby('track_id').apply(lambda x: np.mean(x['distance_speckles_um']))
summary_measurements['mean_intensity'] = aggregated_df.groupby('track_id').apply(lambda x: np.mean(x['mean_intensity']))

sns.relplot(data=summary_measurements, x='mean_distance_ns', y='fano_factor', s=150, hue='track_id', palette='tab20')
ax.set(ylabel="distance from nuclear speckle border ($\mu m)$",
       xlabel='time (s)')
plt.show()



sns.lineplot(data=aggregated_df, x='time_seconds', y='mean_intensity', hue='track_id', palette='tab20', linewidth=3)
plt.show()