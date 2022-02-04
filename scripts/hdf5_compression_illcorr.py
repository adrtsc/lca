import h5py
import numpy as np
from pathlib import Path
from skimage import io
from natsort import natsorted
import re
import cv2
import yaml
import sys

# define the site this job should process
site = int(sys.argv[1])

# load settings
settings_path = Path(sys.argv[2])
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

# get all paths from settings
microscope = settings['microscope']
img_path = Path(settings['paths']['img_path'])
illcorr_path = Path(settings['paths']['illcorr_path'])
output_path = Path(settings['paths']['hdf5_path'])
magnification = settings['magnification']
file_extension = settings['file_extension']
illumination_correction = settings['illumination_correction']
illumination_correction = settings['illumination_correction']

img_files = img_path.glob('*.%s' % file_extension)
img_files = [fyle for fyle in img_files]

if microscope == 'visiscope':

    channel_names = np.unique(
        [re.search("(?<=_w[0-9]).*(?=_)",
                   str(fyle)).group(0) for fyle in img_files])


    # iterate over channels and timepoints and save into dataset
    if any([bool(re.search('(?<=_s)[0-9]{1,}', str(fyle))) for fyle in
            img_files]):
        site_files = img_path.glob('*_s%d_t[0-9]*.%s' % (site, file_extension))
    else:
        site_files = img_path.glob('*.%s' % file_extension)

    site_files = [str(fyle) for fyle in site_files]
    site_files = natsorted(site_files)

    # wells = ['A01']

elif microscope == 'cv7k':
    channel_names = np.unique(
        [re.search(f"(?<=Z[0-9]{{2}}).*(?=.{file_extension})",
                   str(fyle)).group(0) for fyle in img_files])

    # wells = np.unique(
    #     [re.search("(?<=_)[a-zA-Z]+[0-9]{2}(?=_T[0-9]{4})",
    #                str(fyle)).group(0) for fyle in img_files])

    # iterate over channels and timepoints and save into dataset
    site_files = [f for f in img_files if re.search(f'_T[0-9]{{4}}F{site:03d}',
                                                    str(f))]
    site_files = natsorted(site_files)

# pre-load the illumination correction files:
illum_corr = {key: [] for key in channel_names}
illcorr_files = sorted(illcorr_path.glob('*.png'))

for fyle in illcorr_files:
    for channel in channel_names:
        if channel in str(fyle):
            img = io.imread(fyle)
            illum_corr[channel].append(img)


channel_data = {key: [] for key in channel_names}

for fyle in site_files:
    for channel in channel_data:
        if channel in str(fyle):
            if file_extension in ['tif', 'stk']:
                img = io.imread(fyle, plugin="tifffile")
            else:
                img = io.imread(fyle)

            if illumination_correction:
                corrected_image = (cv2.subtract(img, illum_corr[channel][0])) / (illum_corr[channel][1]/np.max(illum_corr[channel][1]))
                channel_data[channel].append(corrected_image.astype('uint16'))
            else:
                channel_data[channel].append(img)


# Open the experiment HDF5 file in "append" mode

with h5py.File(output_path.joinpath(f'site_{site:04d}.hdf5'), "w") as file:

    chunk = list(np.shape(np.squeeze(np.stack(
        channel_data[channel_names[0]]))))
    chunk[0] = 1
    chunk = tuple(chunk)

    grp = file.create_group("intensity_images")

    for channel in channel_data:
        # Create a dataset in the file
        dataset = grp.create_dataset(
            channel, np.shape(np.squeeze(np.stack(channel_data[channel]))),
            data=np.stack(channel_data[channel]),
            compression='gzip', chunks=chunk, shuffle=True,
            fletcher32=True, dtype='uint16')

        dataset.attrs.create(name="element_size_um",
                             data=(1, 6.5/magnification, 6.5/magnification))


