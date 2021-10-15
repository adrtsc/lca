import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
from lca.ndt.segmentation import segment_nuclei_cellpose_2DT, segment_cells_cellpose_2DT
from lca.nd.util import get_label_difference_2D
from lca.ndt.util import find_boundaries_2DT

# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])

nuclei_diameter = 3*settings['magnification']
cell_diameter = 5*settings['magnification']

nuclei_channel = settings['objects']['nuclei']['segmentation_channel']
cell_channel = settings['objects']['cells']['segmentation_channel']

##############################################################################
# end of user settings - no need to change anything beyond this point
##############################################################################


# open the hdf5 file corresponding to this site
file = h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a")

nuclei_images = file['intensity_images/%s' % nuclei_channel][:]
cell_images = file['intensity_images/%s' % cell_channel][:]


nuclei = segment_nuclei_cellpose_2DT(nuclei_images, diameter=nuclei_diameter)
cells = segment_cells_cellpose_2DT(cells_intensity_images=cell_images,
                                   nuclei_intensity_images=nuclei_images,
                                   diameter=cell_diameter)

# get cytoplasm (difference of cells and nuclei)

cytoplasm = get_label_difference_2D(cells, nuclei)

objects = {'nuclei': nuclei,
           'cells': cells,
           'cytoplasm': cytoplasm}


# prepare some things for saving
chunk = list(np.shape(nuclei_images))
chunk[0] = 1
chunk = tuple(chunk)

element_size_um = file['intensity_images/%s' % nuclei_channel].attrs.get('element_size_um')


# create new group for label images and create dataset for each object
grp = file.create_group("label_images")

for object in objects:

    # Create a dataset in the file
    dataset = grp.create_dataset(object, np.shape(objects[object]), h5py.h5t.STD_U16BE, data=objects[object],
                                 compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

    dataset.attrs.create(name="element_size_um", data=element_size_um)


# create new group for boundaries and create dataset for each object
grp = file.create_group("boundary_images")

for object in objects:

    # find boundaries
    boundaries = find_boundaries_2DT(objects[object])

    # Create a dataset in the file
    dataset = grp.create_dataset('%s_boundaries' % object, np.shape(objects[object]), h5py.h5t.STD_U16BE,
                                 data=boundaries, compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

    dataset.attrs.create(name="element_size_um", data=element_size_um)

file.close()

