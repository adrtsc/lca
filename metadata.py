import skimage.measure
import pandas as pd

def measure_metadata(label_images):

    object_regionprops = pd.DataFrame()

    # measure regionprops for each timepoint

    features = ('label', 'bbox',)

    for idx, label_image in enumerate(list(label_images)):

        current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                             properties=features))

        # assess which objects touch the border
        current_regionprops['is_border'] = measure_border_cells(current_regionprops)

        # add timepoint
        current_regionprops['timepoint'] = idx

        object_regionprops = pd.concat([object_regionprops, current_regionprops])

    return object_regionprops