import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
from lca.ndt.segmentation import segment_nuclei_cellpose_3DT


# define the site this job should process
site = 1

# load settings
settings_path = Path('/home/adrian/mnt/work/PhD/Code/Python/lca/scripts/settings/20211111_gpu_settings.yml')

with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

hdf5_path = Path(settings['paths']['hdf5_path'])

objs = {}

for obj in settings['cellpose'].keys():

    if obj == 'nuclei':
        nuclei_channel = settings['cellpose']['nuclei']['channel']

        with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:
            nuclei_images = file['intensity_images/%s' % nuclei_channel][:]

        nuclei = segment_nuclei_cellpose_3DT(
            nuclei_images,
            diameter=settings['cellpose']['nuclei']['diameter'],
            resample=settings['cellpose']['nuclei']['resample'],
            cellprob_threshold=settings['cellpose']['nuclei']['cellprob_threshold'],
            flow_threshold=settings['cellpose']['nuclei']['flow_threshold'],
            do_3D=settings['cellpose']['nuclei']['do_3D'],
            min_size=settings['cellpose']['nuclei']['min_size'],
            apply_filter=settings['cellpose']['nuclei']['apply_filter'],
            anisotropy=settings['scaling'],
            gpu=settings['cellpose']['nuclei']['gpu'])

        objs['nuclei'] = nuclei


with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:

    ref = list(file['intensity_images/'].keys())[0]
    ref_img = file['intensity_images/%s' % ref][:]
    element_size_um = file['intensity_images/%s' % ref].attrs.get(
        'element_size_um')
    chunk = list(np.shape(ref_img))
    chunk[0] = 1
    chunk = tuple(chunk)


    # create new group for label images and create dataset for each obj
    grp = file.create_group("label_images")

    for obj in objs:

        # Create a dataset in the file
        dataset = grp.create_dataset(obj, np.shape(objs[obj]),
                                     h5py.h5t.STD_U16BE, data=objs[obj],
                                     compression='gzip', chunks=chunk,
                                     shuffle=True,fletcher32=True)

        dataset.attrs.create(name="element_size_um", data=element_size_um)
