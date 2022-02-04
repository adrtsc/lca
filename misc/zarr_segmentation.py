import zarr
import sys
from pathlib import Path
import numpy as np
from lca.nd.segmentation import segment_nuclei_cellpose_3D

img_path = Path("/data/active/atschan/lattice_light_sheet/zarr/20211118_hiPSC_day-06-Deskewed.zarr")
img_path = Path(r"Z:\20211111_hiPSC_MS2\GFP_5_RFP_15_4s\zarr\site_0001.zarr")
z = zarr.open(img_path)

keys = [key for key in z.intensity_images.keys()]

out_shape = z['intensity_images'][keys[0]].shape
out_chunks = z['intensity_images'][keys[0]].chunks

#layer = 'layer_02'
#timepoint = int(sys.argv[1])


lbl_grp = z.create_group('label_images')
d = lbl_grp.create_dataset('nuclei',
                           shape=out_shape,
                           chunks=out_chunks,
                           dtype='uint16')

for timepoint in range(100):
    img = z['intensity_images']['sdcRFP590-JF549'][timepoint, :, :, :]

    labels = segment_nuclei_cellpose_3D(intensity_image=img,
                                        diameter=180,
                                        gpu=True,
                                        min_size=10,
                                        anisotropy=500/65,
                                        cellprob_threshold=0,
                                        flow_threshold=27,
                                        apply_filter=True,
                                        filter_sigma=10,
                                        resample=True)


    d[timepoint, :, :, :] = labels