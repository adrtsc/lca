import skimage.measure
import pandas as pd

def measure_intensity(label_images, intensity_images, channel_names):

    object_regionprops = pd.DataFrame()

    # iterate over all channels for intensity measurements

    for channel_id, channel in enumerate(channel_names):

        channel_regionprops = pd.DataFrame()

        features = ('max_intensity', 'mean_intensity', 'min_intensity')

        for idx, label_image in enumerate(list(label_images)):

            intensity_image = intensity_images[channel_id][idx, :, :]
            current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                                 intensity_image=intensity_image,
                                                                                 properties=features))

            feature_names = list(current_regionprops.keys())
            current_regionprops.columns = [t + '_%s' % channel for t in feature_names]

            channel_regionprops = pd.concat([channel_regionprops, current_regionprops])

        object_regionprops = pd.concat([object_regionprops, channel_regionprops], axis=1)

    return object_regionprops

