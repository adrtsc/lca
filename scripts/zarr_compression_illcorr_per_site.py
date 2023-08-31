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


def correct_illumination(img, img_path, illcorr_path, channel, microscope):

    if microscope == 'cv7k':

        illum_files = list(img_path.joinpath('stuff').glob('*.png'))
        dark_files = natsorted(
            [fyle for fyle in illum_files if "DC_sCMOS" in str(fyle)])
        illum_files = natsorted(
            [fyle for fyle in illum_files if "SC_BP" in str(fyle)])


        ch_idx = int(''.join(filter(str.isdigit, channel)))

        dark_channel = io.imread(dark_files[0])
        flat_channel = io.imread(illum_files[ch_idx])
        dark = np.repeat(dark_channel[np.newaxis, :, :], img.shape[0], axis=0)
        flat = flat_channel / np.max(flat_channel)

        corrected_img = cv2.subtract(img, dark) / flat

    elif microscope == 'visiscope':
        # get dark image of the channel
        dark_files = illcorr_path.joinpath('dark').glob('*.png')
        dark_channel = [fyle for fyle in dark_files if channel in str(fyle)]
        dark_image = io.imread(dark_channel[0])
        
        # get illumination image of the channel
        illum_files = illcorr_path.joinpath('illumination').glob('*.png')
        illum_channel = [fyle for fyle in illum_files if channel in str(fyle)]
        illum_image = io.imread(illum_channel[0])
        
        # adjust illum image so that max value is 1
        illum_image = illum_image / np.max(illum_image)
        
        # adjust dimensions
        dark_image = np.repeat(dark_image[np.newaxis, :, :], img.shape[0], axis=0)
        illum_image = np.repeat(illum_image[np.newaxis, :, :], img.shape[0], axis=0)

        corrected_img = cv2.subtract(img, dark_image) / illum_image
        corrected_img = corrected_img.astype('uint16')

    return corrected_img


def get_cv7k_stack(site_files, channel):
    img_stack = []
    channel_files = [fyle for fyle in site_files if channel in str(fyle)]
    for fyle in channel_files:
        img_stack.append(io.imread(fyle))
    img = np.stack(img_stack)
    return img

def get_visiscope_stack(site_files, channel):
    channel_files = [fyle for fyle in site_files if channel in str(fyle)]
    img = io.imread(channel_files[0], plugin='tifffile')
    return img

def get_wells(img_files, microscope):
    if microscope == 'cv7k':
        wells = np.unique(
            [re.search("[a-zA-Z][0-9]{2}(?=_)",
                       str(fyle)).group(0) for fyle in img_files])
    elif microscope == 'visiscope':
        wells = np.unique(
            [re.search("_s[0-9]{1,}_t",
                       str(fyle)).group(0) for fyle in img_files])
        
    return natsorted(wells)
    
def get_image_files(img_path, file_extension, site, microscope):
    
    img_files = img_path.glob(f'*{file_extension}')
    
    if microscope == 'cv7k':
        img_files = [fyle for fyle in img_files if f'F{site:03d}L' in str(fyle)]
    elif microscope == 'visiscope':
        img_files = [fyle for fyle in img_files if f'_t{site}.stk' in str(fyle)]
        
    return img_files
        

def main():
    # define the site this job should process
    site = int(sys.argv[1])

    # load settings
    settings_path = Path(sys.argv[2])
    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)

    # get all paths from settings
    img_path = Path(settings['paths']['img_path'])
    illcorr_path = Path(settings['paths']['illcorr_path'])
    output_path = Path(settings['paths']['zarr_path'])
    file_extension = settings['file_extension']
    illumination_correction = settings['illumination_correction']
    microscope = settings['microscope']
    
    # get the image files of the current timepoint
    img_files = get_image_files(img_path, file_extension, site,  microscope)
    
    # get the wells
    wells = get_wells(img_files, microscope)

    # iterate over channels and wells and save into dataset

    for idx, well in enumerate(wells):
        z = zarr.open(output_path.joinpath(f'Assayplate.zarr'), mode='a')
        print(well)
        well_files = natsorted(
            [fyle for fyle in img_files if well in str(fyle)])
        for channel in z[f'{well}/{site}/intensity_images'].keys():
            print(channel)
            if microscope == 'cv7k':
                img = get_cv7k_stack(well_files, channel)
            elif microscope == 'visiscope':
                img = get_visiscope_stack(site_files, channel)

            if illumination_correction:
                img = correct_illumination(img, img_path, illcorr_path, channel, microscope)

            pyramid = tuple(pyramid_gaussian(img,
                                             max_layer=3,
                                             downscale=2,
                                             preserve_range=True,
                                             channel_axis=0))

            for ilvl, level in enumerate(pyramid):
                z[f'{well}/{site}/intensity_images/{channel}/level_{ilvl:02d}'][:, :,
                :] = np.expand_dims(level, 0)


if __name__ == "__main__":
    main()