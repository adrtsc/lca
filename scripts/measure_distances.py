import numpy as np
import pandas as pd
import sys
from scipy.spatial import KDTree
from pathlib import Path
import zarr
import yaml

def main():

    # define the site this job should process
    site = int(sys.argv[1])

    # load settings
    settings_path = Path(sys.argv[2])
    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)


    zarr_path = Path(settings['paths']['zarr_path'])
    feature_path = Path(settings['paths']['feature_path'])
    z = zarr.open(zarr_path.joinpath(f'site_{site:04d}.zarr'), mode='a')
    speckles = z['label_images/nuclear_speckles/level_00'][:]

    spacing = z['label_images/nuclear_speckles/level_00'].attrs['element_size_um']

    anisotropy = spacing[0]/spacing[1]

    df = pd.read_csv(feature_path.joinpath("20220224_fv_preprocessed_with_mock.csv"))

    df = df[df.site == (site-1)]

    new_df = measure_distance_points_mask_3DT(df, speckles, anisotropy)

    new_df.to_csv(
        feature_path.joinpath(f'distance_measurements/fv_site_{site:04d}.csv'))

def measure_distance_points_mask_3DT(df, mask, anisotropy):
    timepoints = np.unique(df['timepoint'])
    output = []

    for timepoint in timepoints:
        current_mask = mask[timepoint, :, :, :]
        current_df = df.loc[df['timepoint'] == timepoint]

        current_df['centroid-0_blobs'] = np.round(
            current_df['centroid-0_blobs'] * anisotropy)

        seg_coords = pd.DataFrame()
        seg_coords['z'], seg_coords['y'], seg_coords['x'] = np.where(
            current_mask == 1)
        seg_coords['z'] = np.round(seg_coords['z'] * anisotropy)

        tree = KDTree(seg_coords)
        dist, points = tree.query(
            current_df[['centroid-0_blobs',
                        'centroid-1_blobs',
                        'centroid-2_blobs']], 1)
        current_df['distance_object'] = dist

        # same for distance to background (or how deep in a speckle they are)
        bg_coords = pd.DataFrame()
        bg_coords['z'], bg_coords['y'], bg_coords['x'] = np.where(
            current_mask == 0)
        bg_coords['z'] = np.round(bg_coords['z'] * anisotropy)

        tree_bg = KDTree(bg_coords)
        dist, points = tree_bg.query(
            current_df[['centroid-0_blobs',
                        'centroid-1_blobs',
                        'centroid-2_blobs']], 1)
        current_df['distance_bg'] = dist
        output.append(current_df)

        print(f'processed timepoint {timepoint}')

    return pd.concat(output)

if __name__ == "__main__":
   main()