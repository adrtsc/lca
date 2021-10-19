from lca.nd.segmentation import segment_nuclei_cellpose, segment_cells_cellpose
import numpy as np


def segment_nuclei_cellpose_2DT(intensity_images, diameter, resample=True,
                                flow_threshold=0.4, cellprob_threshold=0, gpu=False, torch=False, apply_filter=True):

    label_images = np.zeros(np.shape(intensity_images))

    for idx, intensity_image in enumerate(list(intensity_images)):

        label_image = segment_nuclei_cellpose(intensity_image=intensity_image,
                                              diameter=diameter,
                                              resample=resample,
                                              flow_threshold=flow_threshold,
                                              cellprob_threshold=cellprob_threshold,
                                              gpu=gpu,
                                              torch=torch,
                                              apply_filter=apply_filter)

        label_images[idx, :, :] = label_image

    return label_images


def segment_cells_cellpose_2DT(cells_intensity_images, nuclei_intensity_images,  diameter, resample=True,
                               flow_threshold=0.4, cellprob_threshold=0, gpu=False, torch=False, apply_filter=True):

    label_images = np.zeros(np.shape(cells_intensity_images))

    for idx, cells_intensity_image in enumerate(list(cells_intensity_images)):
        nuclei_intensity_image = nuclei_intensity_images[idx, :, :]
        label_image = segment_cells_cellpose(cells_intensity_image=cells_intensity_image,
                                             nuclei_intensity_image=nuclei_intensity_image,
                                             diameter=diameter,
                                             resample=resample,
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold,
                                             gpu=gpu,
                                             torch=torch,
                                             apply_filter=apply_filter)

        label_images[idx, :, :] = label_image

    return label_images

