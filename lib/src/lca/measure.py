import pandas as pd
import skimage.measure
from lca.utils import measure_border_cells
from skimage.feature import blob_log
from lca.LapTracker import LapTracker

def measure_metadata(label_images):

    object_regionprops = pd.DataFrame()

    # measure regionprops for each timepoint

    features = ('label', 'bbox', 'centroid')

    for idx, label_image in enumerate(list(label_images)):
        current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))

        # assess which objects touch the border
        current_regionprops['is_border'] = measure_border_cells(current_regionprops)

        # add timepoint
        current_regionprops['timepoint'] = idx

        object_regionprops = pd.concat([object_regionprops, current_regionprops])

    return object_regionprops


def measure_morphology(label_images):

    object_regionprops = pd.DataFrame()

    # measure regionprops for each timepoint

    features = ('area', 'bbox', 'bbox_area', 'centroid',
                'convex_area', 'eccentricity', 'equivalent_diameter',
                'euler_number', 'extent', 'major_axis_length',
                'minor_axis_length', 'moments', 'moments_central',
                'moments_hu', 'moments_normalized', 'orientation',
                'perimeter', 'solidity')

    for idx, label_image in enumerate(list(label_images)):
        current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))
        object_regionprops = pd.concat([object_regionprops, current_regionprops])

    return object_regionprops


def measure_intensity(label_images, intensity_images):

    object_regionprops = pd.DataFrame()

    features = ('max_intensity', 'mean_intensity', 'min_intensity')

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                             intensity_image=intensity_image,
                                                                             properties=features))
        object_regionprops = pd.concat([object_regionprops, current_regionprops])

    return object_regionprops


def measure_blobs(label_images, intensity_images,
                  min_sigma=5,
                  max_sigma=10,
                  num_sigma=1,
                  threshold=0.001,
                  overlap=0.5,
                  exclude_border=True):
    blobs = pd.DataFrame()

    for idx, label_image in enumerate(list(label_images)):
        intensity_image = intensity_images[idx, :, :]
        current_blobs = blob_log(intensity_image,
                                 min_sigma=min_sigma,
                                 max_sigma=max_sigma,
                                 num_sigma=num_sigma,
                                 threshold=threshold,
                                 overlap=overlap,
                                 exclude_border=exclude_border)

        current_blobs = pd.DataFrame(current_blobs, columns=['centroid-0',
                                                             'centroid-1',
                                                             'size'])
        object_labels = []
        for index, blob in current_blobs.iterrows():
            current_label = label_image[int(blob['centroid-0']), int(blob['centroid-1'])]
            object_labels.append(current_label)

        current_blobs['label'] = object_labels
        current_blobs['timepoint'] = idx
        current_blobs = current_blobs.loc[current_blobs['label'] > 0]
        blobs = pd.concat([blobs, current_blobs])

    return blobs


def measure_tracks(metadata):

    metadata = metadata.reset_index()
    tracks = pd.DataFrame()
    tracks['unique_object_id'] = metadata['unique_object_id']

    tracker = LapTracker(max_distance=30, time_window=3, max_split_distance=100,
                         max_gap_closing_distance=100)

    columns = ['centroid-0', 'centroid-1', 'timepoint', 'label']
    tracks['track_id'] = tracker.track_df(metadata, identifiers=columns)

    return tracks