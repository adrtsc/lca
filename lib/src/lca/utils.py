import numpy as np


def measure_border_cells(feature_values):
    # check which cells are border cells
    border_1 = feature_values['bbox-0'] == 0
    border_2 = feature_values['bbox-1'] == 0
    border_3 = feature_values['bbox-2'] == 2048
    border_4 = feature_values['bbox-3'] == 2048
    borders = border_1 | border_2 | border_3 | border_4
    borders = np.array(borders, dtype=int)

    return borders

def measure_assignment(label_images, assigned_label_images, md, md_assignment):

    assigned_unique_object_ids = []

    for idx, label_image in enumerate(list(label_images)):

        current_md = md_assignment.loc[md_assignment['timepoint'] == idx]
        assigned_label_image = assigned_label_images[idx, :, :]
        assigned_label = get_parent_labels(label_image, assigned_label_image)

        assigned_unique_object_id = []

        for assignment in assigned_label:
            current_unique_object_id = np.array(current_md.loc[current_md.label == assignment]['unique_object_id'])[0]
            assigned_unique_object_id.append(current_unique_object_id)

        assigned_unique_object_ids.append(assigned_unique_object_id)

    assigned_unique_object_ids = [item for sublist in assigned_unique_object_ids for item in sublist]

    md['unique_object_id'] = assigned_unique_object_ids
    md.set_index('unique_object_id')

    return md


def get_parent_labels(label_img_1, label_img_2):
    labels = np.unique(label_img_1)
    labels = labels[labels.nonzero()]

    assigned_labels = []

    for label in labels:

        values = label_img_2[label_img_1 == label]
        values = values[values.nonzero()]

        assigned_label = np.unique(values)

        if len(assigned_label) > 1:
            raise ValueError('object to be assigned extends over multiple objects in second label image')
        if len(assigned_label) == 0:
            raise ValueError('object to be assigned is not contained in any object in second label image')
        elif len(assigned_label) == 1:
            assigned_labels.append(assigned_label[0])

    return assigned_labels