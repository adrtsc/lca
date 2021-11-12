from skimage.io import imread
from dask import delayed
import dask.array as da
from natsort import natsorted
from glob import glob
import napari
import numpy as np
import re

filenames = natsorted(glob(
    r"C:\Users\Adrian\Desktop\GFP_5_RFP_15_Cy5_15_6s\*.stk"))

channels = np.unique([re.search("(?<=_w[0-9]).*(?=_)",
                                str(fyle)).group(0) for fyle in filenames])
colors = ['blue', 'green', 'gray']


# read the first file to get the shape and dtype
# ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
sample = imread(filenames[0], plugin='tifffile')
viewer = napari.Viewer()


for idx, channel in enumerate(channels):
    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn, plugin='tifffile') for fn in filenames if channel in fn]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
    stack.shape  # (nfiles, nz, ny, nx)

    viewer.add_image(stack, colormap=colors[idx], contrast_limits=(105, 500),
                     scale=[500 / 65, 1, 1], multiscale=False)


# compare to having the arrays in memory:

for idx, channel in enumerate(channels):

    array = [imread(fn, plugin='tifffile') for fn in filenames if channel in fn]
    # Stack into one large dask.array
    stack = np.stack(array, axis=0)

    viewer.add_image(stack, colormap=colors[idx], contrast_limits=(105, 500),
                     scale=[500 / 65, 1, 1], multiscale=False)


