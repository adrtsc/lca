import zarr
import sys
from pathlib import Path
from lca.nd.segmentation import segment_nuclei_cellpose_3D

img_path = Path("/data/active/atschan/lattice_light_sheet/zarr/20211118_hiPSC_day-06-Deskewed.zarr")
z = zarr.open(img_path)
layer = 'layer_02'
timepoint = int(sys.argv[1])
img = z['intensity_images']['561'][layer][timepoint, :, :, :]

labels = segment_nuclei_cellpose_3D(intensity_image=img,
                                    diameter=20,
                                    gpu=True,
                                    min_size=10,
                                    anisotropy=200/144,
                                    cellprob_threshold=0,
                                    flow_threshold=27,
                                    apply_filter=True,
                                    filter_sigma=2.5,
                                    resample=True)

lbl_grp = z.create_group('label_images')
nuc_grp = lbl_grp.create_group('nuclei')
layer_grp = nuc_grp.create_group(layer)
d = layer_grp.create_dataset(layer,
                             shape=np.insert(np.shape(labels), 0, 120),
                             chunks=[29, 223, 512],
                             dtype='uint16')

d[timepoint, :, :, :] = labels