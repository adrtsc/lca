import pandas as pd
import numpy as np
import functools
from lca.ndt.LapTracker import LapTracker
from lca.nd.measure import measure_morphology_2D
from lca.nd.measure import measure_intensity_2D
from lca.nd.measure import measure_blobs_2D


def iterate_time(func):
    @functools.wraps(func)
    def wrapper_decorator(**kwargs):

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


measure_morphology_2DT = iterate_time(measure_morphology_2D)
measure_intensity_2DT = iterate_time(measure_intensity_2D)
measure_blobs_2DT = iterate_time(measure_blobs_2D)





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
