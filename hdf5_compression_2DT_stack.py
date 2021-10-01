import sys
import h5py
import numpy as np
from pathlib import Path
from skimage import io


img_path = Path(r'Z:\20210928_dummy')
output_path = Path(r'Z:\20210928_dummy\hdf5')

n_sites = 2
sites = ['_s%d.stk' % site for site in np.arange(1, n_sites+1)]

for idx, site in enumerate(sites):

    img_files = img_path.glob('*%s' % site)
    channels = {'sdcRFP590-JF549': [], 'sdcGFP': []}

    for fyle in img_files:
        for channel in channels:
            if channel in str(fyle):
                img = io.imread(fyle, plugin="tifffile")
                channels[channel].append(img)

    # Open the experiment HDF5 file in "append" mode

    file = h5py.File(output_path.joinpath('site_%d.hdf5' %idx), "w")
    chunk = list(np.shape(np.squeeze(np.stack(channels[channel]))))
    chunk[0] = 1
    chunk = tuple(chunk)

    for channel in channels:
        # Create a dataset in the file
        dataset = file.create_dataset(
            channel, np.shape(np.squeeze(np.stack(channels[channel]))),
            h5py.h5t.STD_U16BE, data=np.stack(channels[channel]),
            compression='gzip', chunks=chunk, shuffle=True,
            fletcher32=True)

    file.close()