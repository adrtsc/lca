import sys
import h5py
import numpy as np
from pathlib import Path
from skimage import io


img_path = Path(r'Z:\20210928_dummy')
output_path = Path(r'Z:\20210928_dummy\hdf5')

n_sites = 1
sites = np.arange(1, n_sites+1)

for idx, site in enumerate(sites):

    img_files = img_path.glob("*_s%d_t[0-9].stk" % site)
    channels = {'sdcRFP590-JF549': [], 'sdcGFP': []}

    for fyle in img_files:
        for channel in channels:
            if channel in str(fyle):
                img = io.imread(fyle, plugin="tifffile")
                channels[channel].append(img)


    data = np.stack(channels[channel])

    for timepoint in range(0, data.shape[0]):
        current_data = data[timepoint, :, :, :]
        # Open the experiment HDF5 file in "append" mode
        file = h5py.File(output_path.joinpath('site_%d_timepoint_%d.hdf5' % (idx, timepoint)), "w")
        chunk = list(np.shape(current_data))
        chunk[0] = 1
        chunk = tuple(chunk)


        for channel in channels:
            # Create a dataset in the file
            dataset = file.create_dataset(
                channel, np.shape(current_data),
                h5py.h5t.STD_U16BE, data=current_data,
                compression='gzip', chunks=chunk, shuffle=True,
                fletcher32=True)

    file.close()