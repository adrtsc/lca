import skimage.measure
import pandas as pd

def measure_morphology(file):

    # iterate over all objects and measure their features
    regionprops = pd.DataFrame()

    for oid, object in enumerate(file['label_images'].keys()):

        object_regionprops = pd.DataFrame()

        # measure regionprops for each timepoint

        features = ('label', 'area', 'bbox', 'bbox_area', 'centroid',
                    'convex_area', 'eccentricity', 'equivalent_diameter',
                    'euler_number', 'extent', 'major_axis_length',
                    'minor_axis_length', 'moments', 'moments_central',
                    'moments_hu', 'moments_normalized', 'orientation',
                    'perimeter', 'solidity')

        for idx, label_image in enumerate(list(file['label_images/%s' % object])):

            current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                                 properties=features))
            feature_names = list(current_regionprops.keys())
            current_regionprops.columns = [ t + '_%s' % object for t in feature_names]

            if oid == 0:
                current_regionprops['timepoint'] = idx

            object_regionprops = pd.concat([object_regionprops, current_regionprops])

        regionprops = pd.concat([regionprops, object_regionprops],axis=1, sort=False)

    return regionprops