from lca.nd.segmentation import segment_nuclei_cellpose_2D
from lca.nd.segmentation import segment_nuclei_cellpose_3D
from lca.nd.segmentation import segment_cells_cellpose
import numpy as np
from skimage.segmentation import find_boundaries


def segment_nuclei_cellpose_2DT(intensity_images, diameter, resample=True,
                                flow_threshold=0.4, cellprob_threshold=0,
                                min_size=2500, gpu=False, torch=False,
                                apply_filter=True, do_3D=False,
                                anisotropy=1.0, **kwargs):

    label_images = []

    for idx, intensity_image in enumerate(list(intensity_images)):

        label_image = segment_nuclei_cellpose_2D(
            intensity_image=intensity_image,
            diameter=diameter,
            resample=resample,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D,
            min_size=min_size,
            gpu=gpu,
            torch=torch,
            apply_filter=apply_filter,
            anisotropy=anisotropy)

        label_images.append(label_image.astype('uint16'))

    label_images = np.stack(label_images, axis=0)

    return label_images


def segment_nuclei_cellpose_3DT(intensity_images, diameter, resample=False,
                                flow_threshold=0.4, cellprob_threshold=0,
                                min_size=2500, gpu=False, torch=False,
                                apply_filter=True, do_3D=True,
                                anisotropy=1.0, **kwargs):

    label_images = []

    for idx, intensity_image in enumerate(list(intensity_images)):

        label_image = segment_nuclei_cellpose_3D(
            intensity_image=intensity_image,
            diameter=diameter,
            resample=resample,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D,
            min_size=min_size,
            gpu=gpu,
            torch=torch,
            apply_filter=apply_filter,
            anisotropy=anisotropy)

        label_images.append(label_image.astype('uint16'))

        print(f'processed timepoint {idx}.')

    label_images = np.stack(label_images, axis=0)

    return label_images


def segment_cells_cellpose_2DT(cells_intensity_images,
                               nuclei_intensity_images=None, model='cyto',
                               diameter=100, resample=True, flow_threshold=0.4,
                               cellprob_threshold=0, min_size=10000, gpu=False,
                               torch=False, apply_filter=True, **kwargs):

    label_images = []

    for idx, cells_intensity_image in enumerate(list(cells_intensity_images)):
        if nuclei_intensity_images is not None:
            nuclei_intensity_image = nuclei_intensity_images[idx, :, :]
        else:
            nuclei_intensity_image = None
        label_image = segment_cells_cellpose(
            cells_intensity_image=cells_intensity_image,
            nuclei_intensity_image=nuclei_intensity_image,
            model='cyto',
            diameter=diameter,
            resample=resample,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            gpu=gpu,
            torch=torch,
            apply_filter=apply_filter)

        label_images.append(label_image.astype('uint16'))

    label_images = np.stack(label_images, axis=0)

    return label_images


def find_boundaries_2DT(label_images):

    boundaries = np.zeros(np.shape(label_images))

    for idx, lbl in enumerate(list(label_images)):
        cb = find_boundaries(lbl)
        boundaries[idx, :, :] = cb

    return boundaries
