import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
from lca.ndt.segmentation import segment_nuclei_cellpose_2DT, segment_cells_cellpose_2DT, find_boundaries_2DT
from lca.nd.util import get_label_difference_2D


# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])

with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])

objs = {}

for obj in settings['cellpose'].keys():

    if obj == 'nuclei':
        nuclei_channel = settings['cellpose']['nuclei']['channel']

        with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:
            nuclei_images = file['intensity_images/%s' % nuclei_channel][:]

        nuclei = segment_nuclei_cellpose_2DT(nuclei_images,
                                             diameter=settings['cellpose']['nuclei']['diameter'],
                                             resample=settings['cellpose']['nuclei']['resample'],
                                             cellprob_threshold=settings['cellpose']['nuclei']['cellprob_threshold'],
                                             flow_threshold=settings['cellpose']['nuclei']['flow_threshold'],
                                             do_3D=settings['cellpose']['nuclei']['do_3D'],
                                             min_size=settings['cellpose']['nuclei']['min_size'],
                                             apply_filter=settings['cellpose']['nuclei']['apply_filter'],
                                             anisotropy=settings['scaling'],
                                             gpu=False)

        objs['nuclei'] = nuclei

    elif obj == 'cells':

        cells_channel = settings['cellpose']['cells']['channel']


        if settings['cellpose']['cells']['nuclei_channel']:

            nuclei_channel = settings['cellpose']['cells']['nuclei_channel']

            with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:
                cell_images = file['intensity_images/%s' % cells_channel][:]
                nuclei_images = file['intensity_images/%s' % nuclei_channel][:]
        else:

            with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:
                cell_images = file['intensity_images/%s' % cells_channel][:]
                nuclei_images = None

        cells = segment_cells_cellpose_2DT(cells_intensity_images=cell_images,
                                           nuclei_intensity_images=nuclei_images,
                                           model=settings['cellpose']['cells']['model'],
                                           diameter=settings['cellpose']['cells']['diameter'],
                                           resample=settings['cellpose']['cells']['resample'],
                                           cellprob_threshold=settings['cellpose']['cells']['cellprob_threshold'],
                                           flow_threshold=settings['cellpose']['cells']['flow_threshold'],
                                           min_size=settings['cellpose']['cells']['min_size'])

        objs['cells'] = cells





if settings['cellpose']['cytoplasm']:
    # get cytoplasm (difference of cells and nuclei)
    cytoplasm = get_label_difference_2D(cells, nuclei)
    objs['cytoplasm'] = cytoplasm




with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:

    ref = list(file['intensity_images/'].keys())[0]
    ref_img = file['intensity_images/%s' % ref][:]
    element_size_um = file['intensity_images/%s' % ref].attrs.get('element_size_um')
    chunk = list(np.shape(ref_img))
    chunk[0] = 1
    chunk = tuple(chunk)


    # create new group for label images and create dataset for each obj
    grp = file.create_group("label_images")

    for obj in objs:

        # Create a dataset in the file
        dataset = grp.create_dataset(obj, np.shape(objs[obj]), h5py.h5t.STD_U16BE, data=objs[obj],
                                     compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

        dataset.attrs.create(name="element_size_um", data=element_size_um)


    # create new group for boundaries and create dataset for each obj
    grp = file.create_group("boundary_images")

    for obj in objs:

        # find boundaries
        boundaries = find_boundaries_2DT(objs[obj])

        # Create a dataset in the file
        dataset = grp.create_dataset('%s_boundaries' % obj, np.shape(objs[obj]), h5py.h5t.STD_U16BE,
                                     data=boundaries, compression='gzip', chunks=chunk, shuffle=True,fletcher32=True)

        dataset.attrs.create(name="element_size_um", data=element_size_um)



