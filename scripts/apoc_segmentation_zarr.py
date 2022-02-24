import zarr
from skimage.transform import resize
from pathlib import Path
import yaml
import sys
import apoc

# define the site this job should process
timepoint = int(sys.argv[1])

print(timepoint)

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

#level = 'level_00'

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

cl_filename = Path(r"/data/active/atschan/20220218_hiPSC_MS2/apoc/20220218_classifier.cl")
classifier = apoc.PixelClassifier(cl_filename)

for fyle in zarr_files:

    z = zarr.open(fyle, mode='a')
    levels = [key for key in z['intensity_images']['sdc-GFP']]
    keys = [key for key in z.intensity_images.keys()]

    for level in levels:

        out_shape = z['intensity_images'][keys[0]][level].shape
        out_chunks = z['intensity_images'][keys[0]][level].chunks
        output_spacing = z['intensity_images'][keys[0]][level].attrs['element_size_um']

        img = z['intensity_images']['sdc-GFP'][level][timepoint, :, :, :]

        result = classifier.predict(image=img)
        # subtract 1 from resulting label image (label 1 is background label)
        result = result - 1

        if not hasattr(z, f'label_images/nuclear_speckles/{level}'):
            d = z.create_dataset(f'label_images/nuclear_speckles/{level}',
                                 shape=out_shape,
                                 chunks=out_chunks,
                                 dtype='uint16')

            d.attrs["element_size_um"] = output_spacing

        z[f'label_images/nuclear_speckles/{level}'][timepoint, :, :, :] = result

    '''# resize the label image for other levels

    levels = [key for key in z['intensity_images']['sdc-GFP']]
    levels.remove(level)

    for lvl in levels:

        output_shape = z[f'intensity_images/sdc-GFP/{lvl}'][timepoint, :, :, :].shape
        out_shape = z['intensity_images'][keys[0]][lvl].shape
        out_chunks = z['intensity_images'][keys[0]][lvl].chunks

        resized = resize(image=result,
                         output_shape=output_shape,
                         anti_aliasing=True,
                         preserve_range=True,
                         anti_aliasing_sigma= 5,
                         order=0)

        if not hasattr(z, f'label_images/nuclear_speckles/{lvl}'):
            z.create_dataset(f'label_images/nuclear_speckles/{lvl}',
                             shape=out_shape,
                             chunks=out_chunks,
                             dtype='uint16')

        z[f'label_images/nuclear_speckles/{lvl}'][timepoint, :, :, :] = resized'''



