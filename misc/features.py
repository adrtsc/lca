import skimage.measure
from skimage.feature import blob_log
import pandas as pd
from lca.utils import measure_border_cells


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

        features = ('area', 'bbox', 'bbox_area', 'centroid',
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


class Blobs(Features):

    def __init__(self, label_images, intensity_images):
        self.label_images = label_images
        self.intensity_images = intensity_images

    def extract(self, min_sigma=5, max_sigma=10, num_sigma=1, threshold=0.001, overlap=0.5, exclude_border=True):

        blobs = pd.DataFrame()

        for idx, label_image in enumerate(list(self.label_images)):
            intensity_image = self.intensity_images[idx, :, :]
            current_blobs = blob_log(intensity_image,
                                     min_sigma=min_sigma,
                                     max_sigma=max_sigma,
                                     num_sigma=num_sigma,
                                     threshold=threshold,
                                     overlap=overlap,
                                     exclude_border=exclude_border)

            current_blobs = pd.DataFrame(current_blobs, columns=['y_coordinates',
                                                                 'x_coordinates',
                                                                 'size'])
            object_labels = []
            for index, blob in current_blobs.iterrows():
                current_label = label_image[int(blob['y_coordinates']), int(blob['x_coordinates'])]
                object_labels.append(current_label)

            current_blobs['label'] = object_labels
            current_blobs = current_blobs.loc[current_blobs['label'] > 0]
            blobs = pd.concat([blobs, current_blobs])

        return blobs