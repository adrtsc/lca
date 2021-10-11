import skimage.measure
import pandas as pd
import numpy as np
from lca.util import measure_border_cells
from skimage.feature import blob_log
from skimage.draw import disk



def measure_metadata_2D(label_image):

    features = ('label', 'bbox', 'centroid')
    regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))

    # assess which objects touch the border
    regionprops['is_border'] = measure_border_cells(regionprops)

    return regionprops


def measure_morphology_2D(label_image):

    # measure regionprops for each timepoint
    features = ('area', 'bbox', 'bbox_area', 'centroid',
                'convex_area', 'eccentricity', 'equivalent_diameter',
                'euler_number', 'extent', 'major_axis_length',
                'minor_axis_length', 'moments', 'moments_central',
                'moments_hu', 'moments_normalized', 'orientation',
                'perimeter', 'solidity')

    regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))

    return regionprops


def measure_intensity_2D(label_image, intensity_image):

    features = ('max_intensity', 'mean_intensity', 'min_intensity')
    regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                 intensity_image=intensity_image,
                                                                 properties=features))


    return regionprops


def measure_blobs_2D(intensity_image,
                     label_image=None,
                     min_sigma=5,
                     max_sigma=10,
                     num_sigma=1,
                     threshold=0.001,
                     overlap=0.5,
                     exclude_border=True):

    blobs = blob_log(intensity_image,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     num_sigma=num_sigma,
                     threshold=threshold,
                     overlap=overlap,
                     exclude_border=exclude_border)

    blobs = pd.DataFrame(blobs, columns=['centroid-0',
                                         'centroid-1',
                                         'size'])

    if label_image is not None:
        object_labels = []

        for index, blob in blobs.iterrows():
            current_label = label_image[int(blob['centroid-0']), int(blob['centroid-1'])]
            object_labels.append(current_label)

        blobs['label'] = object_labels
        blobs = blobs.loc[blobs['label'] > 0]

    min_intensity = []
    max_intensity = []
    mean_intensity = []
    var_intensity = []
    mean_bg_intensity = []

    for index, row in blobs.iterrows():

        rr, cc = disk(tuple(row[['centroid-0', 'centroid-1']]), row['size'],
                      shape=np.shape(intensity_image))

        rr_bg, cc_bg = disk(tuple(row[['centroid-0', 'centroid-1']]),
                            2 * row['size'],
                            shape=np.shape(intensity_image))

        pixels = intensity_image[rr, cc]
        pixels_bg = intensity_image[rr_bg, cc_bg]

        n_pixels = len(pixels)
        n_pixels_bg = len(pixels_bg)

        mean_bg_intensity.append((np.sum(pixels_bg) - np.sum(pixels))
                                 / (n_pixels_bg - n_pixels))

        mean_intensity.append(np.mean(pixels))

        min_intensity.append(np.min(pixels))
        max_intensity.append(np.max(pixels))
        var_intensity.append(np.var(pixels))


    blobs['min_intensity'] = min_intensity
    blobs['max_intensity'] = max_intensity
    blobs['mean_intensity'] = mean_intensity
    blobs['var_intensity'] = var_intensity
    blobs['mean_background_intensity'] = mean_bg_intensity
    blobs['SNR'] = np.array(mean_intensity) / np.array(mean_bg_intensity)

    return blobs


