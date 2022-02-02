import pandas as pd
import functools
from lca.ndt.LapTracker import LapTracker
from lca.nd.measure import measure_morphology_2D
from lca.nd.measure import measure_intensity_2D
from lca.nd.measure import measure_blobs_2D


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
