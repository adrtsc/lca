import pandas as pd
from numpy.random import uniform
from lca.ndt.LapTracker import LapTracker
from lca.nd.measure import measure_morphology_2D, measure_intensity_2D, measure_blobs_2D


def measure_morphology_2DT(label_images):

    regionprops = []

    # measure regionprops for each timepoine
    for idx, label_image in enumerate(list(label_images)):
        current_regionprops = measure_morphology_2D(label_image)
        current_regionprops['timepoint'] = idx
        regionprops.append(current_regionprops)

    regionprops = pd.concat(regionprops)

    return regionprops


def measure_intensity_2DT(label_images, intensity_images):

    regionprops = []

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_regionprops = measure_intensity_2D(label_image, intensity_image)
        current_regionprops['timepoint'] = idx
        regionprops.append(current_regionprops)

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


def measure_tracks_2DT(df, max_distance,
                       time_window,
                       max_split_distance,
                       max_gap_closing_distance,
                       modulate_centroids=True):

    tracker = LapTracker(max_distance=max_distance, time_window=time_window, max_split_distance=max_split_distance,
                         max_gap_closing_distance=max_gap_closing_distance)


    columns = ['centroid-0', 'centroid-1', 'timepoint', 'label']
    df['track_id'] = tracker.track_df(df, identifiers=columns, modulate_centroids=modulate_centroids)

    return df


