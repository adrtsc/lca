import zarr
from pathlib import Path
import yaml
import sys
import apoc

# define the site this job should process
timepoint = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

level = 'level_00'

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

cl_filename = Path(r"Z:\20220218_hiPSC_MS2\apoc\20220218_classifier.cl")
classifier = apoc.PixelClassifier(cl_filename)

for fyle in zarr_files:

    z = zarr.open(fyle, mode='a')

    keys = [key for key in z.intensity_images.keys()]

    out_shape = z['intensity_images'][keys[0]][level].shape
    out_chunks = z['intensity_images'][keys[0]][level].chunks

    img = z['intensity_images']['sdc-GFP'][level][timepoint, :, :, :]

    result = classifier.predict(image=img)

    if not hasattr(z, 'label_images'):
        z.create_group('label_images')


    d = z['label_images'].create_dataset('nuclear_speckles',
                                         shape=out_shape,
                                         chunks=out_chunks,
                                         dtype='uint16')

    z['label_images']['nuclear_speckles'][timepoint, :, :, :] = result
