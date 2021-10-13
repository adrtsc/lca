import numpy as np
from lca.nd.util import get_parent_labels_2D
from skimage.segmentation import find_boundaries

def measure_assignment_2DT(label_images, assigned_label_images, fv, fv_assignment):

    assigned_unique_object_ids = []

    for idx, label_image in enumerate(list(label_images)):

        current_fv = fv.loc[fv['timepoint'] == idx]
        assigned_label_image = assigned_label_images[idx, :, :]
        assigned_label = get_parent_labels_2D(assigned_label_image, label_image)
        print(len(assigned_label))

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


def find_boundaries_2DT(label_images):

    boundaries = np.zeros(np.shape(label_images))

    for idx, lbl in enumerate(list(label_images)):
        cb = find_boundaries(lbl)
        boundaries[idx, :, :] = cb

    return boundaries
