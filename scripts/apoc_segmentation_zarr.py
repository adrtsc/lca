import zarr
from skimage.transform import resize
from pathlib import Path
import yaml
import sys
import apoc

# define the site this job should process
timepoint = int(sys.argv[1])

print(timepoint)

channel = 'C02'

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

cl_filename = Path(r"/data/active/atschan/20220414_hiPSC_MS2_CV7K/apoc/20220414.cl")
classifier = apoc.PixelClassifier(cl_filename)

for fyle in zarr_files:
    print(fyle)

    z = zarr.open(fyle, mode='a')
    levels = [key for key in z['intensity_images'][channel]]
    keys = [key for key in z.intensity_images.keys()]

    for level in levels:

        out_shape = z['intensity_images'][keys[0]][level].shape
        out_chunks = z['intensity_images'][keys[0]][level].chunks
        output_spacing = z['intensity_images'][keys[0]][level].attrs['element_size_um']

        img = z['intensity_images'][channel][level][timepoint, :, :, :]

        result = classifier.predict(image=img)
        # subtract 1 from resulting label image (label 1 is background label)
        result = result - 1

        if not hasattr(z, f'label_images/nuclear_speckles/{level}'):
            z.create_dataset(f'label_images/nuclear_speckles/{level}',
                             shape=out_shape,
                             chunks=out_chunks,
                             dtype='uint16')

        z[f'label_images/nuclear_speckles/{level}'].attrs["element_size_um"] = output_spacing
        z[f'label_images/nuclear_speckles/{level}'][timepoint, :, :, :] = result
