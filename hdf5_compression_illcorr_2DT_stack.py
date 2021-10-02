import h5py
import numpy as np
from pathlib import Path
from skimage import io
import re
import sys

img_path = Path(r'Z:\20210930_dummy\MIP')
illcorr_path = Path(r'Z:\illumination_correction')
output_path = Path(r'Z:\20210930_dummy\hdf5')

img_files = img_path.glob('*.tif')
img_files = [fyle for fyle in img_files]
sites = np.unique([int(re.search("(?<=_s)[0-9]{1,}", str(fyle)).group(0)) for fyle in img_files])
channel_names = np.unique([re.search("(?<=_w[0-9])[^\W_]+(?=_s)", str(fyle)).group(0) for fyle in img_files])

# define the site this job should process
site = int(sys.argv[1])

# pre-load the illumination correction files:

illum_corr = {key: [] for key in channel_names}
illcorr_files = illcorr_path.glob('*.png')

for fyle in illcorr_files:
    for channel in channel_names:
        if channel in str(fyle):
            img = io.imread(fyle)
            illum_corr[channel].append(img)


# iterate over channels and timepoints and save into dataset
site_files = img_path.glob('*_s%d_t[0-9].tif' % site)
channel_data = {key: [] for key in channel_names}

for fyle in site_files:
    for channel in channel_data:
        if channel in str(fyle):
            img = io.imread(fyle, plugin="tifffile")
            #corrected_image = (cv2.subtract(img, illum_corr[channel][0])) / (illum_corr[channel][1]/np.max(illum_corr[channel][1]))
            corrected_image = img
            channel_data[channel].append(corrected_image)

# Open the experiment HDF5 file in "append" mode

file = h5py.File(output_path.joinpath('site_%04d.hdf5' % site), "w")
chunk = list(np.shape(np.squeeze(np.stack(channel_data[channel]))))
chunk[0] = 1
chunk = tuple(chunk)

grp = file.create_group("intensity_images")

for channel in channel_data:
    # Create a dataset in the file
    dataset = grp.create_dataset(
        channel, np.shape(np.squeeze(np.stack(channel_data[channel]))),
        h5py.h5t.STD_U16BE, data=np.stack(channel_data[channel]),
        compression='gzip', chunks=chunk, shuffle=True,
        fletcher32=True)

    dataset.attrs.create(name="element_size_um", data=(1, 0.1625, 0.1625))


file.close()
