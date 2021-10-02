def measure_assignment(label_images, assigned_label_images, md, md_assignment):

    assigned_unique_ids = []

    for idx, label_image in enumerate(list(label_images)):

        current_md = md_assignment.loc[md_assignment['timepoint'] == idx]
        assigned_label_image = assigned_label_images[idx, :, :]
        assigned_label = get_parent_labels(label_image, assigned_label_image)

        assigned_unique_id = []

        for assignment in assigned_label:
            current_unique_id = np.array(current_md.loc[current_md.label == assignment]['unique_id'])[0]
            assigned_unique_id.append(current_unique_id)

        assigned_unique_ids.append(assigned_unique_id)

    assigned_unique_ids = [item for sublist in assigned_unique_ids for item in sublist]

    md['unique_id'] = assigned_unique_ids
    md.set_index('unique_id')

    return md