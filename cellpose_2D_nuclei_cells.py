import sys
from cellpose import models
from skimage.measure import label
from pathlib import Path
import h5py
import numpy as np
from lca.utils import get_label_difference

# user settings
path_to_images = Path(r'/data/active/atschan/20210930_dummy/hdf5/')
nuclei_diameter = 90
cell_diameter = 250

nuclei_channel = 'sdcDAPIxmRFPm'
cell_channel = 'sdcDAPIxmRFPm'

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

nuclei_images = file['intensity_images/%s' % nuclei_channel]
cell_images = file['intensity_images/%s' % cell_channel]



# initialize arrays for label images
nuclei = np.zeros(np.shape(nuclei_images))
cells = np.zeros(np.shape(cell_images))

# run cellpose on the specified images of the specified channel
model_nuc = models.Cellpose(gpu=False, model_type="nuclei")
model_cells = models.Cellpose(gpu=False, model_type="cyto2")

for idx, nuclei_image in enumerate(list(nuclei_images)):

    nuclear_masks, flows, styles, diams = model_nuc.eval(nuclei_image,
                                                         channels=[0, 0],
                                                         resample=True,
                                                         diameter=nuclei_diameter,
                                                         flow_threshold=flow_threshold,
                                                         cellprob_threshold=cellprob_threshold)

    cell_image = cell_images[idx, :, :]
    cell_image = np.stack([cell_image, nuclei_image])
    cells_masks, flows, styles, diams = model_cells.eval(cell_image,
                                                         channels=[1, 2],
                                                         resample=True,
                                                         diameter=cell_diameter,
                                                         flow_threshold=flow_threshold,
                                                         cellprob_threshold=cellprob_threshold)

    # relabel for safety
    nuclei[idx, :, :] = label(nuclear_masks).astype('uint16')
    cells[idx, :, :] = label(cells_masks).astype('uint16')

# get cytoplasm (difference of cells and nuclei)

cytoplasm = get_label_difference(cells, nuclei)

# create new group for label images and create dataset for nuclei
grp = file.create_group("label_images")
chunk = list(np.shape(nuclei_images))
chunk[0] = 1
chunk = tuple(chunk)

# Create a dataset in the file
dataset = grp.create_dataset('nuclei', np.shape(nuclei), h5py.h5t.STD_U16BE, data=nuclei,
                             compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

dataset.attrs.create(name="element_size_um", data=(1, 0.1625, 0.1625))

dataset = grp.create_dataset('cells', np.shape(cells), h5py.h5t.STD_U16BE, data=cells,
                             compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

dataset.attrs.create(name="element_size_um", data=(1, 0.1625, 0.1625))

dataset = grp.create_dataset('cytoplasm', np.shape(cells), h5py.h5t.STD_U16BE, data=cells,
                             compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

dataset.attrs.create(name="element_size_um", data=(1, 0.1625, 0.1625))

file.close()