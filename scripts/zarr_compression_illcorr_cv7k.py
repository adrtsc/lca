import zarr
import numpy as np
from pathlib import Path
from skimage import io
from natsort import natsorted
from skimage.transform import pyramid_gaussian
import re
import cv2
import yaml
import sys


def correct_illumination(img, illum_files, dark_files, channel):
    ch_idx = int(''.join(filter(str.isdigit, channel)))

    dark_channel = io.imread(dark_files[0])
    flat_channel = io.imread(illum_files[ch_idx])
    dark = np.repeat(dark_channel[np.newaxis, :, :], img.shape[0], axis=0)
    flat = flat_channel / np.max(flat_channel)

    corrected_img = cv2.subtract(img, dark) / flat

    return corrected_img.astype('uint16')

# define the tp this job should process
tp = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# get all paths from settings
img_path = Path(settings['paths']['img_path'])
illcorr_path = Path(settings['paths']['illcorr_path'])
output_path = Path(settings['paths']['zarr_path'])
mag = settings['magnification']
file_extension = settings['file_extension']
illumination_correction = settings['illumination_correction']

img_files = img_path.glob(f'*.{file_extension}')
img_files = [fyle for fyle in img_files if f'T{tp:04d}' in str(fyle)]

illum_files = img_path.joinpath('stuff').glob('*.tif')
illum_files = [fyle for fyle in illum_files]

channel_names = np.unique(
    [re.search("C[0-9]{2}(?=.tif)",
               str(fyle)).group(0) for fyle in img_files])
sites = np.unique(
    [re.search("F[0-9]{3}(?=L)",
               str(fyle)).group(0) for fyle in img_files])
n_slices = len(np.unique(
    [re.search("Z[0-9]{2,}(?=C)",
               str(fyle)).group(0) for fyle in img_files]))


illum = natsorted([fyle for fyle in illum_files if "SC_BP" in str(fyle)])
dark = natsorted([fyle for fyle in illum_files if "DC_sCMOS" in str(fyle)])


# iterate over channels and sites and save into dataset

for idx, site in enumerate(sites):
    z = zarr.open(output_path.joinpath(f'site_{idx+1:04d}.zarr'), mode='a')
    print(site)
    site_files = natsorted(
        [fyle for fyle in img_files if site in str(fyle)])
    for channel in channel_names:
        print(channel)
        img_stack = []
        channel_files = [fyle for fyle in site_files if channel in str(fyle)]
        for fyle in channel_files:
            img_stack.append(io.imread(fyle))

        img = np.stack(img_stack)

        if illumination_correction:
            img = correct_illumination(img, illum, dark, channel)

        pyramid = tuple(pyramid_gaussian(img,
                                         max_layer=3,
                                         downscale=2,
                                         preserve_range=True,
                                         channel_axis=0))

        for ilvl, level in enumerate(pyramid):
            z[f'intensity_images/{channel}/level_{ilvl:02d}'][tp-1, :, :,
            :] = level





