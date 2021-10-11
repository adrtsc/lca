from cellpose import models
import scipy.ndimage as ndi
from skimage.measure import label

def segment_nuclei_cellpose(intensity_image, diameter, resample=True, flow_threshold=0.4,
                            cellprob_threshold=0, gpu=True, torch=False, filter=True):

    model = models.Cellpose(gpu=gpu, torch=torch, model_type="nuclei")

    if filter:
        intensity_image = ndi.median_filter(intensity_image, 10)

    label_image, flows, styles, diams = model.eval(intensity_image,
                                                   channels=[0, 0],
                                                   resample=resample,
                                                   diameter=diameter,
                                                   flow_threshold=flow_threshold,
                                                   cellprob_threshold=cellprob_threshold)
    # relabel for safety
    label_image = label(label_image).astype('uint16')

    return label_image

def segment_cells_cellpose(cells_intensity_image, nuclei_intensity_image,  diameter, resample=True, flow_threshold=0.4,
                           cellprob_threshold=0, gpu=False, torch=False, filter=True):

    model = models.Cellpose(gpu=gpu, torch=torch, model_type="cyto2")

    if filter:
        cells_intensity_image = ndi.median_filter(cells_intensity_image, 10)
        nuclei_intensity_image = ndi.median_filter(nuclei_intensity_image, 10)

    merged_intensity_image = np.stack([cells_intensity_image, nuclei_intensity_image, nuclei_intensity_image], axis=2)

    label_image, flows, styles, diams = model.eval(merged_intensity_image,
                                                   channels=[0, 0],
                                                   resample=resample,
                                                   diameter=diameter,
                                                   flow_threshold=flow_threshold,
                                                   cellprob_threshold=cellprob_threshold)

    # relabel for safety
    label_image = label(label_image).astype('uint16')

    return label_image
