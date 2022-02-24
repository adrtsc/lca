import zarr
import sys
from pathlib import Path
from lca.nd.segmentation import segment_nuclei_cellpose_3D


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

    img = z['intensity_images']['sdcRFP590-JF549'][level][timepoint, :, :, :]

    labels = segment_nuclei_cellpose_3D(intensity_image=img,
                                        diameter=120,
                                        gpu=False,
                                        min_size=10,
                                        anisotropy=9.25,
                                        cellprob_threshold=0,
                                        flow_threshold=27,
                                        apply_filter=True,
                                        filter_sigma=10,
                                        resample=True)


    if not hasattr(z, 'label_images'):
        z.create_group('label_images')

    d = z['label_images'].create_dataset('nuclei',
                                         shape=out_shape,
                                         chunks=out_chunks,
                                         dtype='uint16')

    d[timepoint, :, :, :] = labels

