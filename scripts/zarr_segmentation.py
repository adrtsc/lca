import zarr
import sys
import yaml
from pathlib import Path
from skimage.transform import resize
from lca.nd.segmentation import segment_nuclei_cellpose_3D


# define the site this job should process
timepoint = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# define which level should be used for segmentation
level = settings['cellpose']['nuclei']['level']

# define which channel should be used for segmentation
channel = settings['cellpose']['nuclei']['channel']

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]


for fyle in zarr_files:

    z = zarr.open(fyle, mode='a')

    keys = [key for key in z.intensity_images.keys()]

    output_spacing = z[f'intensity_images/{channel}/{level}'].attrs['element_size_um']
    out_shape = z['intensity_images'][keys[0]][level].shape
    out_chunks = z['intensity_images'][keys[0]][level].chunks

    z_res = z['intensity_images'][channel][level].attrs['element_size_um'][0]
    y_res = z['intensity_images'][channel][level].attrs['element_size_um'][1]
    anisotropy = z_res/y_res

    img = z['intensity_images'][channel][level][timepoint, :, :, :]

    labels = segment_nuclei_cellpose_3D(intensity_image=img,
                                        anisotropy=anisotropy,
                                        **settings['cellpose']['nuclei'])

    if not hasattr(z, f'label_images/nuclei/{level}'):
        z.create_dataset(f'label_images/nuclei/{level}',
                         shape=out_shape,
                         chunks=out_chunks,
                         dtype='uint16')

    z[f'label_images/nuclei/{level}'][timepoint, :, :, :] = labels
    z[f'label_images/nuclei/{level}'].attrs["element_size_um"] = output_spacing

    # resize the label image for other levels

    levels = [key for key in z['intensity_images'][channel]]
    levels.remove(level)

    for lvl in levels:

        output_shape = z[f'intensity_images/{channel}/{lvl}'][timepoint, :, :, :].shape
        output_spacing = z[f'intensity_images/{channel}/{lvl}'].attrs['element_size_um']
        out_shape = z['intensity_images'][keys[0]][lvl].shape
        out_chunks = z['intensity_images'][keys[0]][lvl].chunks

        resized = resize(image=labels,
                         output_shape=output_shape,
                         preserve_range=True,
                         order=0)

        if not hasattr(z, f'label_images/nuclei/{lvl}'):
            z.create_dataset(f'label_images/nuclei/{lvl}',
                             shape=out_shape,
                             chunks=out_chunks,
                             dtype='uint16')

        z[f'label_images/nuclei/{lvl}'].attrs["element_size_um"] = output_spacing
        z[f'label_images/nuclei/{lvl}'][timepoint, :, :, :] = resized


