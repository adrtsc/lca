import numpy as np
import pandas as pd
import warnings
import skimage.measure

def get_parent_labels_2D(label_img, parent_label_img):

    centroids = pd.DataFrame(
        skimage.measure.regionprops_table(label_img,
                                          properties=('label', 'centroid')))
    assigned_labels = []

    for id, label in centroids.iterrows():

        assigned_label = parent_label_img[int(label['centroid-0']),
                                          int(label['centroid-1'])]

        if assigned_label == 0:
            warnings.warn('object to be assigned is not contained in any object in second label image and will be lost in the aggregated dataframe')
            assigned_labels.append(np.nan)
        elif assigned_label > 0:
            assigned_labels.append(assigned_label)

    return assigned_labels


def get_label_difference_2D(label_1, label_2):
    A = np.copy(label_1)
    A[label_2 > 0] = 0

    return A

