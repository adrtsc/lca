from skimage import io
from pathlib import Path
from skimage.measure import label
import numpy as np

path_to_ns = Path(r"Z:\20210930_dummy\temp_segmentations")
                  
ns = io.imread(path_to_ns.joinpath('nuclear_speckles_s1.tif'), plugin="tifffile")
labeled = np.zeros(np.shape(ns), dtype='uint16')

for t in range(0, np.shape(ns)[0]):
    labeled[t, :, :] = label(ns[t, :, :])

io.imsave(path_to_ns.joinpath('nuclear_speckles_labeled_s1.tif'), labeled, plugin="tifffile")

file['label_images/nuclear_speckles'] = labeled