import skimage.measure
import pandas as pd

def measure_morphology(label_images):

    object_regionprops = pd.DataFrame()

    # measure regionprops for each timepoint

    features = ('label', 'area', 'bbox', 'bbox_area', 'centroid',
                'convex_area', 'eccentricity', 'equivalent_diameter',
                'euler_number', 'extent', 'major_axis_length',
                'minor_axis_length', 'moments', 'moments_central',
                'moments_hu', 'moments_normalized', 'orientation',
                'perimeter', 'solidity')

    for idx, label_image in enumerate(list(label_images)):

        current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))

        #feature_names = list(current_regionprops.keys())
        #current_regionprops.columns = [ t + '_%s' % object for t in feature_names]

        object_regionprops = pd.concat([object_regionprops, current_regionprops])

    return object_regionprops