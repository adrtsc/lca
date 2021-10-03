import skimage.measure
import pandas as pd
from lca.measure_border_cells import measure_border_cells


class Features():

    def __init__(self, label_image, intensity_image=None):
        self.label_image = label_image
        self.intensity_image = intensity_image

    def extract(self):
        pass


class Morphology(Features):

    def __init__(self, label_images):
        self.label_images = label_images


    def extract(self):

        object_regionprops = pd.DataFrame()

        # measure regionprops for each timepoint

        features = ('label', 'area', 'bbox', 'bbox_area', 'centroid',
                    'convex_area', 'eccentricity', 'equivalent_diameter',
                    'euler_number', 'extent', 'major_axis_length',
                    'minor_axis_length', 'moments', 'moments_central',
                    'moments_hu', 'moments_normalized', 'orientation',
                    'perimeter', 'solidity')

        for idx, label_image in enumerate(list(self.label_images)):
            current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))
            object_regionprops = pd.concat([object_regionprops, current_regionprops])

        return object_regionprops


class Intensity(Features):

    def __init__(self, label_images, intensity_images=None):
        self.label_images = label_images
        self.intensity_images = intensity_images

    def extract(self):

        object_regionprops = pd.DataFrame()

        features = ('max_intensity', 'mean_intensity', 'min_intensity')

        for idx, label_image in enumerate(list(self.label_images)):
            intensity_image = self.intensity_images[idx, :, :]
            current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                                 intensity_image=intensity_image,
                                                                                 properties=features))
            object_regionprops = pd.concat([object_regionprops, current_regionprops])

        return object_regionprops

'''
put the time measurement into the metadata for the moment

class Time(Features):

    def __init__(self, label_images, intensity_images=None):
        self.label_images = label_image
        self.intensity_images = intensity_image

    def extract(self):

        object_regionprops = pd.DataFrame()

            for idx, label_image in enumerate(list(self.label_images)):
                current_regionprops = pd.DataFrame()
                current_regionprops['timepoint'] = [idx]*len(np.unique(label_image[label_image.nonzero()]))
                object_regionprops = pd.concat([object_regionprops, current_regionprops])

            return object_regionprops
'''

class Metadata(Features):

    def __init__(self, label_images):
        self.label_images = label_images

    def extract(self):

        object_regionprops = pd.DataFrame()

        # measure regionprops for each timepoint

        features = ('label', 'bbox', 'centroid')

        for idx, label_image in enumerate(list(self.label_images)):
            current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=features))

            # assess which objects touch the border
            current_regionprops['is_border'] = measure_border_cells(current_regionprops)

            # add timepoint
            current_regionprops['timepoint'] = idx

            object_regionprops = pd.concat([object_regionprops, current_regionprops])

        return object_regionprops