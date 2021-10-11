import numpy as np
from lca.nd.util import get_parent_labels_2D
from skimage.segmentation import find_boundaries

def measure_assignment_2DT(label_images, assigned_label_images, md, md_assignment):

    assigned_unique_object_ids = []
    aggregate = False

    md = md.reset_index()

    for idx, label_image in enumerate(list(label_images)):

        current_md = md.loc[md['timepoint'] == idx]
        assigned_label_image = assigned_label_images[idx, :, :]
        assigned_label = get_parent_labels_2D(assigned_label_image, label_image)

        assigned_unique_object_id = []

        for assignment in assigned_label:
            if np.isnan(assignment):
                current_unique_object_id = np.nan
            else:
                current_unique_object_id = np.array(current_md.loc[current_md.label == assignment]['unique_object_id'])[0]

            assigned_unique_object_id.append(current_unique_object_id)

        assigned_unique_object_ids.append(assigned_unique_object_id)

    assigned_unique_object_ids = [item for sublist in assigned_unique_object_ids for item in sublist]

    md_assignment['unique_object_id'] = assigned_unique_object_ids
    md_assignment.set_index('unique_object_id')

    return md_assignment


def find_boundaries_2DT(label_images):

    boundaries = np.zeros(np.shape(label_images))

    for idx, lbl in enumerate(list(label_images)):
        cb = find_boundaries(lbl)
        boundaries[idx, :, :] = cb

    return boundaries
