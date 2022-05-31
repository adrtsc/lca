import numpy as np
import warnings
import skimage.measure
from lca.nd.util import get_parent_labels_2D
import pandas as pd
import skimage

def measure_assignment_2DT(label_images, assigned_label_images, fv, fv_assignment):

    assigned_unique_object_ids = []

    for idx, label_image in enumerate(list(label_images)):

        current_fv = fv.loc[fv['timepoint'] == idx]
        assigned_label_image = assigned_label_images[idx, :, :]
        assigned_label = get_parent_labels_2D(assigned_label_image, label_image)
        assigned_unique_object_id = []

        for assignment in assigned_label:
            if np.isnan(assignment):
                current_unique_object_id = np.nan
            else:
                current_unique_object_id = np.array(current_fv.loc[current_fv.label == assignment]['unique_object_id'])[0]

            assigned_unique_object_id.append(current_unique_object_id)

        assigned_unique_object_ids.append(assigned_unique_object_id)

    assigned_unique_object_ids = [item for sublist in assigned_unique_object_ids for item in sublist]

    fv_assignment['unique_object_id'] = assigned_unique_object_ids
    fv_assignment.set_index('unique_object_id')

    return fv_assignment


def get_parent_labels_2DT(label_images, parent_label_images):

    parent_labels = []

    for idx, label_image in enumerate(label_images):

        parent_label_image = parent_label_images[idx, :, :]
        centroids = pd.DataFrame(skimage.measure.regionprops_table(label_image, properties=('label', 'centroid')))
        assigned_labels = []

        for id, label in centroids.iterrows():

            assigned_label = parent_label_image[int(label['centroid-0']), int(label['centroid-1'])]

            if assigned_label == 0:
                warnings.warn('object to be assigned is not contained in any object in second label image and will be lost in the aggregated dataframe')
                assigned_labels.append(np.nan)
            elif assigned_label > 0:
                assigned_labels.append(assigned_label)

        assigned_labels = pd.DataFrame({'label': centroids['label'],
                                        'timepoint': idx,
                                        'parent_label': assigned_labels})
        parent_labels.append(assigned_labels)

    parent_labels = pd.concat(parent_labels)

    return parent_labels

