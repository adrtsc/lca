import sys
from cellpose import models
from skimage.measure import label
from pathlib import Path


# user settings

path_to_images = Path(r"Z:\20210930_dummy\hdf5")
nuclei_diameter = 90
segmentation_channel = 'sdcDAPIxmRFPm'

# thresholds are at default values, can be changed if that doesn't work well

flow_threshold = 0.4
cellprob_threshold = 0.0

##############################################################################
# end of user settings - no need to change anything beyond this point
##############################################################################

# define the site this job should process

site = int(sys.argv[1])

# open the hdf5 file corresponding to this site

file = h5py.File(path_to_images.joinpath('site_%04d.hdf5' % site), "a")
channel_images = file['intensity_images/%s' % segmentation_channel]
nuclei = np.zeros(np.shape(channel_images))


# run cellpose on the specified images of the specified channel

model_nuc = models.Cellpose(gpu=False, torch=False, model_type="nuclei")


for idx, image in enumerate(list(channel_images)):

    nuclear_masks, flows, styles, diams = model_nuc.eval(
        image,
        channels=[0, 0],
        resample=True,
        diameter=nuclei_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold)

    # relabel for safety
    nuclei[idx, :, :] = label(nuclear_masks).astype('uint16')

# create new group for label images and create dataset for nuclei
grp = file.create_group("label_images")

# Create a dataset in the file
dataset = grp.create_dataset('nuclei', np.shape(nuclei), h5py.h5t.STD_U16BE, data=nuclei,
                             compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

dataset.attrs.create(name="element_size_um", data=(1, 0.1625, 0.1625))

file.close()