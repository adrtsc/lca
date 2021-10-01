import skimage.measure
import pandas as pd

def measure_intensity(file):
    # iterate over all objects and measure their features
    regionprops = pd.DataFrame()

    for object in file['label_images'].keys():

        object_regionprops = pd.DataFrame()

        # iterate over all channels for intensity measurements

        for channel in file['intensity_images'].keys():

            channel_regionprops = pd.DataFrame()

            features = ('max_intensity', 'mean_intensity', 'min_intensity')

            for idx, label_image in enumerate(list(file['label_images/%s' % object])):

                intensity_image = file['intensity_images/%s' % channel][idx, :, :]
                current_regionprops = pd.DataFrame(skimage.measure.regionprops_table(label_image,
                                                                                     intensity_image=intensity_image,
                                                                                     properties=features))

                feature_names = list(current_regionprops.keys())
                current_regionprops.columns = [t + '_%s_%s' % (channel, object) for t in feature_names]

                channel_regionprops = pd.concat([channel_regionprops, current_regionprops])

            object_regionprops = pd.concat([object_regionprops, channel_regionprops], axis=1)


        regionprops = pd.concat([object_regionprops, regionprops], axis=1)

    return regionprops

