import pandas as pd
from lca.ndt.LapTracker import LapTracker
from lca.nd.measure import measure_metadata_2D, measure_morphology_2D, measure_intensity_2D, measure_blobs_2D


def measure_metadata_2DT(label_images):

    regionprops = pd.DataFrame()

    # measure regionprops for each timepoine
    for idx, label_image in enumerate(list(label_images)):
        current_regionprops = measure_metadata_2D(label_image)

        # add timepoint
        current_regionprops['timepoint'] = idx

        regionprops = pd.concat([regionprops, current_regionprops])

    return regionprops


def measure_morphology_2DT(label_images):

    regionprops = pd.DataFrame()

    # measure regionprops for each timepoine
    for idx, label_image in enumerate(list(label_images)):
        current_regionprops = measure_morphology_2D(label_image)
        regionprops = pd.concat([regionprops, current_regionprops])

    return regionprops


def measure_intensity_2DT(label_images, intensity_images):

    regionprops = pd.DataFrame()

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_regionprops = measure_intensity_2D(label_image, intensity_image)
        regionprops = pd.concat([regionprops, current_regionprops])

    return regionprops


def measure_blobs_2DT(label_images, intensity_images,
                  min_sigma=5,
                  max_sigma=10,
                  num_sigma=1,
                  threshold=0.001,
                  overlap=0.5,
                  exclude_border=True):

    blobs = pd.DataFrame()

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_blobs = measure_blobs_2D(label_image, intensity_image,
                                      max_sigma=max_sigma,
                                      min_sigma=min_sigma,
                                      num_sigma=num_sigma,
                                      threshold=threshold,
                                      overlap=overlap,
                                      exclude_border=exclude_border)

        current_blobs['timepoint'] = idx
        blobs = pd.concat([blobs, current_blobs])

    return blobs


def measure_tracks_2DT(metadata):

    metadata = metadata.reset_index()
    tracks = pd.DataFrame()
    tracks['unique_object_id'] = metadata['unique_object_id']

    tracker = LapTracker(max_distance=30, time_window=3, max_split_distance=100,
                         max_gap_closing_distance=100)

    columns = ['centroid-0', 'centroid-1', 'timepoint', 'label']
    tracks['track_id'] = tracker.track_df(metadata, identifiers=columns)

    return tracks


