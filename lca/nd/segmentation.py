from cellpose import models
import scipy.ndimage as ndi
import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects



def segment_nuclei_cellpose(intensity_image, diameter, resample=True, flow_threshold=0.4, min_size=2500,
                            cellprob_threshold=0, gpu=False, torch=False, apply_filter=True, do_3D=False,
                            anisotropy=1.0):

    model = models.Cellpose(gpu=gpu, torch=torch, model_type="nuclei")

    if apply_filter:
        intensity_image = ndi.median_filter(intensity_image, 10)

    label_image, flows, styles, diams = model.eval(intensity_image,
                                                   channels=[0, 0],
                                                   resample=resample,
                                                   diameter=diameter,
                                                   flow_threshold=flow_threshold,
                                                   cellprob_threshold=cellprob_threshold,
                                                   do_3D=do_3D,
                                                   anisotropy=anisotropy)

    # remove small artifacts, "min_size" from cellpose is hard to interpret (does not really correspond to pixel count)
    label_image = remove_small_objects(label_image, min_size)

    # relabel for safety
    label_image = label(label_image).astype('uint16')

    return label_image


def segment_cells_cellpose(cells_intensity_image, nuclei_intensity_image=None, model='cyto',
                           diameter=100, resample=True, flow_threshold=0.4, cellprob_threshold=0, min_size=10000,
                           gpu=False, torch=False, apply_filter=True):

    model = models.Cellpose(gpu=gpu, torch=torch, model_type=model)

    if apply_filter:
        cells_intensity_image = ndi.median_filter(cells_intensity_image, 10)
        if nuclei_intensity_image is not None:
            nuclei_intensity_image = ndi.median_filter(nuclei_intensity_image, 10)

    if nuclei_intensity_image is not None:
        merged_intensity_image = np.stack([cells_intensity_image, nuclei_intensity_image, nuclei_intensity_image], axis=2)
        channels = [1, 2]
    else:
        merged_intensity_image = cells_intensity_image
        channels = [0, 0]


    label_image, flows, styles, diams = model.eval(merged_intensity_image,
                                                   channels=channels,
                                                   resample=resample,
                                                   diameter=diameter,
                                                   flow_threshold=flow_threshold,
                                                   cellprob_threshold=cellprob_threshold)

    # remove small artifacts, "min_size" from cellpose is hard to interpret (does not really correspond to pixel count)
    label_image = remove_small_objects(label_image, min_size)

    # relabel for safety
    label_image = label(label_image).astype('uint16')

    return label_image
