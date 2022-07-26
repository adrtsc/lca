import pandas as pd
import numpy as np
import functools
from lca.ndt.LapTracker import LapTracker
from lca.nd.measure import measure_morphology_2D
from lca.nd.measure import measure_intensity_2D
from lca.nd.measure import measure_coordinates_3D
from lca.nd.measure import measure_blobs_2D
from lca.nd.measure import measure_blobs_3D
from abbott.itk_measure import (get_shape_features_dataframe,
                                get_intensity_features_dataframe)
from abbott.conversions import to_itk


def iterate_time(func):
    @functools.wraps(func)
    def wrapper_decorator(**kwargs):
        """
           Wrapper function to apply measurement function for 2D label images
           on 2DT label images.
        """

        if 'label_image' in kwargs:
            dims = kwargs['label_image'].shape
        else:
            dims = kwargs['intensity_image'].shape

        out = list()
        for t in range(0, dims[0]):
            # give only one slice of the input array as input for function
            relevant_keys = ['label_image', 'intensity_image']
            new_kwargs = {key: value[t, :, :] if key in relevant_keys
            else value for (key, value) in kwargs.items()}
            df = func(**new_kwargs)
            # append timepoint
            df['timepoint'] = t
            # append to output df
            out.append(df)
        return pd.concat(out)
    return wrapper_decorator

add_docstring = (
    '''\n This is a wrapped version of this function.
    It takes a 3D ndarray (t, y, x) as input and will compute the
    measurements for every timepoint. The resulting pandas.DataFrame will 
    contain an additional column ('timepoint') to indicate the timepoint
    of the measurement.''')


measure_morphology_2DT = iterate_time(measure_morphology_2D)
measure_intensity_2DT = iterate_time(measure_intensity_2D)
measure_blobs_2DT = iterate_time(measure_blobs_2D)

measure_morphology_2DT.__doc__ += add_docstring
measure_intensity_2DT.__doc__ += add_docstring
measure_blobs_2DT.__doc__ += add_docstring

def measure_morphology_3DT(label_images, spacing):

    regionprops = []

    # measure regionprops for each timepoine
    for idx, label_image in enumerate(list(label_images)):
        label_image = to_itk(label_image, spacing=spacing)
        current_regionprops = get_shape_features_dataframe(label_image)
        current_regionprops['timepoint'] = idx

        # adjust some of the column names that are relevant for tracking
        current_regionprops.rename(columns={'Centroid_y': 'centroid-0',
                                            'Centroid_x': 'centroid-1',
                                            'Centroid_z': 'centroid-3',
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


def measure_intensity_3DT(label_images, intensity_images, spacing):

    regionprops = []

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]

        label_image = to_itk(label_image, spacing=spacing)
        intensity_image = to_itk(intensity_image, spacing=spacing)

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


def measure_coordinates_3DT(blobs,
                            intensity_images,
                            sigma):
    result = []

    for idx, intensity_image in enumerate(list(intensity_images)):
        print(idx)
        c_blobs = blobs.loc[blobs['timepoint'] == idx]
        intensity_image = intensity_images[idx, :, :, :]
        current_blobs = measure_coordinates_3D(c_blobs,
                                               intensity_image,
                                               sigma)
        result.append(current_blobs)

    blobs = pd.concat(result)

    return blobs


def measure_tracks_2DT(df, max_distance,
                       time_window,
                       max_split_distance,
                       max_gap_closing_distance,
                       allow_splitting=True,
                       allow_merging=False,
                       modulate_centroids=True):

    tracker = LapTracker(max_distance=max_distance,
                         time_window=time_window,
                         max_split_distance=max_split_distance,
                         max_gap_closing_distance=max_gap_closing_distance,
                         allow_splitting=allow_splitting,
                         allow_merging=allow_merging)

    columns = ['centroid-0', 'centroid-1', 'timepoint', 'label']
    df['track_id'] = tracker.track_df(df, identifiers=columns,
                                      modulate_centroids=modulate_centroids)

    return df
