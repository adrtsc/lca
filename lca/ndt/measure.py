import pandas as pd
import numpy as np
from lca.ndt.LapTracker import LapTracker
from lca.nd.measure import measure_morphology_2D
from lca.nd.measure import measure_intensity_2D
from lca.nd.measure import measure_blobs_2D
from lca.nd.measure import measure_blobs_3D
from abbott.itk_measure import (get_shape_features_dataframe,
                                get_intensity_features_dataframe)
from abbott.conversions import to_itk

def measure_morphology_2DT(label_images):

    regionprops = []

    # measure regionprops for each timepoine
    for idx, label_image in enumerate(list(label_images)):
        current_regionprops = measure_morphology_2D(label_image)
        current_regionprops['timepoint'] = idx
        regionprops.append(current_regionprops)

    regionprops = pd.concat(regionprops)

    return regionprops


def measure_morphology_3DT(label_images, spacing):

    regionprops = []
    inv_spacing = spacing.copy()
    inv_spacing.reverse()

    # measure regionprops for each timepoine
    for idx, label_image in enumerate(list(label_images)):
        label_image = to_itk(label_image, spacing=inv_spacing)
        current_regionprops = get_shape_features_dataframe(label_image)
        current_regionprops['timepoint'] = idx

        # adjust some of the column names that are relevant for tracking
        current_regionprops.rename(columns={'Centroid_y': 'centroid-1',
                                            'Centroid_x': 'centroid-2',
                                            'Centroid_z': 'centroid-0',
                                            'BoundingBox_lower_x': 'bbox-2',
                                            'BoundingBox_upper_x': 'bbox-5',
                                            'BoundingBox_lower_y': 'bbox-1',
                                            'BoundingBox_upper_y': 'bbox-4',
                                            'BoundingBox_lower_z': 'bbox-0',
                                            'BoundingBox_upper_z': 'bbox-3',
                                            'Label': 'label'},
                                   inplace=True)

        # measure borders
        border_0, border_1, border_2, border_3 = (0,
                                                  0,
                                                  np.shape(label_image)[-2],
                                                  np.shape(label_image)[-1])

        is_border = []

        for idx, obj in current_regionprops.iterrows():
            # check which cells are border cells
            is_border_0 = obj['bbox-1'] == border_0
            is_border_1 = obj['bbox-2'] == border_1
            is_border_2 = obj['bbox-4'] == border_2
            is_border_3 = obj['bbox-5'] == border_3
            is_border.append(
                is_border_0 | is_border_1 | is_border_2 | is_border_3)

        current_regionprops['is_border'] = is_border

        regionprops.append(current_regionprops)

        print(f'measured morphology features of timepoint {idx}')

    regionprops = pd.concat(regionprops)

    return regionprops


def measure_intensity_2DT(label_images, intensity_images):

    regionprops = []

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_regionprops = measure_intensity_2D(label_image,
                                                   intensity_image)
        current_regionprops['timepoint'] = idx
        regionprops.append(current_regionprops)

    regionprops = pd.concat(regionprops)

    return regionprops


def measure_intensity_3DT(label_images, intensity_images, spacing):

    regionprops = []
    inv_spacing = spacing.copy()
    inv_spacing.reverse()

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]

        label_image = to_itk(label_image, spacing=inv_spacing)
        intensity_image = to_itk(intensity_image, spacing=inv_spacing)

        current_regionprops = get_intensity_features_dataframe(label_image,
                                                               intensity_image)
        current_regionprops['timepoint'] = idx

        # adjust some of the column names that are relevant for tracking
        current_regionprops.rename(columns={'Label': 'label'},
                                   inplace=True)
        regionprops.append(current_regionprops)


        print(f'measured intensity features of timepoint {idx}')

    regionprops = pd.concat(regionprops)

    return regionprops


def measure_blobs_2DT(intensity_images,
                      label_images,
                      min_sigma=5,
                      max_sigma=10,
                      num_sigma=1,
                      threshold=0.001,
                      overlap=0.5,
                      exclude_border=True):

    blobs = pd.DataFrame()

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_blobs = measure_blobs_2D(intensity_image,
                                         label_image,
                                         min_sigma=min_sigma,
                                         max_sigma=max_sigma,
                                         num_sigma=num_sigma,
                                         threshold=threshold,
                                         overlap=overlap,
                                         exclude_border=exclude_border)

        current_blobs['timepoint'] = idx
        blobs = pd.concat([blobs, current_blobs])

    return blobs


def measure_blobs_3DT(intensity_images,
                      label_images,
                      min_sigma=5,
                      max_sigma=10,
                      num_sigma=1,
                      threshold=0.001,
                      overlap=0.5,
                      exclude_border=True):

    blobs = pd.DataFrame()

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_blobs = measure_blobs_3D(intensity_image,
                                         label_image,
                                         min_sigma=min_sigma,
                                         max_sigma=max_sigma,
                                         num_sigma=num_sigma,
                                         threshold=threshold,
                                         overlap=overlap,
                                         exclude_border=exclude_border)

        current_blobs['timepoint'] = idx
        blobs = pd.concat([blobs, current_blobs])

    return blobs


def measure_tracks_2DT(df,
                       max_distance,
                       time_window,
                       max_split_distance,
                       max_gap_closing_distance,
                       allow_splitting=True,
                       allow_merging=False,
                       modulate_centroids=True,
                       centroid_0='centroid-0',
                       centroid_1='centroid-1',
                       ):

    tracker = LapTracker(max_distance=max_distance,
                         time_window=time_window,
                         max_split_distance=max_split_distance,
                         max_gap_closing_distance=max_gap_closing_distance,
                         allow_splitting=allow_splitting,
                         allow_merging=allow_merging)

    columns = [centroid_0, centroid_1, 'timepoint', 'label']
    df['track_id'] = tracker.track_df(df, identifiers=columns,
                                      modulate_centroids=modulate_centroids)

    return df
