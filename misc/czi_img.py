import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from skimage.transform import pyramid_gaussian
import zarr

# define image path and output path
img_path = Path(r"C:\Users\Adrian\Desktop\20211118_hiPSC_day-06-Deskewed.czi")
output_path = Path(r"C:\Users\Adrian\Desktop\lattice_light_sheet\zarr")

# get an AICSImage object
aics_img = AICSImage(img_path)

# load image data into memory
img = aics_img.dask_data  # returns 6D STCZYX numpy array

# check how many channels are present
channel_names = aics_img.channel_names
n_channels = len(channel_names)
channel_list = [img[:, channel, :, :, :] for channel in range(0, n_channels)]

# Open the experiment zarr file to append data
z = zarr.open(output_path.joinpath(f'20211118_hiPSC_day-06-Deskewed.zarr'), mode='w')
chunk = [1, 114, 889, 2048]
chunk = tuple(chunk)

grp = z.create_group("intensity_images")

'''
# iterate over channels and create pyramids
for channel_id, channel in enumerate(channel_list):

     move the timepoint axis to the -1 position. There is a bug in
     pyramid_gaussian that causes the function to break if the invariant axis 
     is not in the -1 position


    channel = np.moveaxis(channel, source=0, destination=3)

    # generate the pyramid
    pyramid = tuple(pyramid_gaussian(channel,
                                     max_layer=3,
                                     downscale=2,
                                     preserve_range=True,
                                     channel_axis=-1))


    # restore original axis order
    pyramid = [np.moveaxis(layer,
                           source=3,
                           destination=0) for layer in pyramid]

    # create group for every channel and save pyramid levels as datasets
    for idx, layer in enumerate(pyramid):

        p_grp = grp.create_group(f'{channel_names[channel_id]}')

        d = p_grp.create_dataset(
            f'layer_{idx:02d}',
            shape=np.shape(layer),
            chunks=chunk,
            dtype='uint16')

        d[:] = layer
'''


# iterate over channels and create pyramids
for channel_id, channel in enumerate(channel_list):

    '''
     move the timepoint axis to the -1 position. There is a bug in
     pyramid_gaussian that causes the function to break if the invariant axis 
     is not in the -1 position
    '''

    # create group for every channel and save pyramid levels as datasets
    p_grp = grp.create_group(f'{channel_names[channel_id]}')

    tp_list = list(channel)

    for tp_idx, timepoint in enumerate(tp_list):

        # generate the pyramid
        pyramid = tuple(pyramid_gaussian(timepoint,
                                         max_layer=3,
                                         downscale=2,
                                         preserve_range=True,
                                         channel_axis=None))

            # create group for every channel and save pyramid levels as datasets
            for idx, layer in enumerate(pyramid):
                if tp_idx == 0:
                    p_grp.create_dataset(
                        f'layer_{idx:02d}',
                        shape=np.insert(np.shape(layer), 0, 120),
                        chunks=chunk,
                        dtype='uint16')

                p_grp[f'layer_{idx:02d}'][tp_idx, :, :, :] = layer