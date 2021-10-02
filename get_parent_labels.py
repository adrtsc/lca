import numpy as np

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