import zarr
import numpy as np
from pathlib import Path
import re
import yaml
import sys

IMG_DIMS = {'cv7k': [2160, 2560], 'visiscope': [2048, 2048]}
PIXEL_SIZE_UM = 6.5
N_LEVELS = 4


def extract_metadata_visiscope(file):
    d = {}
    with open(str(file)) as f:
        for line in f:
            try:
                (key, val) = line.split(", ")
                val = val.split("\n")[0]
                if val.isnumeric():
                    val = int(val)
                else:
                    val = val.replace('"', '')
                d[key.replace('"', '')] = val
            except:
                print("extracting key, value failed")
    channels = {key: value for (key, value) in d.items() if 'WaveName' in key}
    channel_names = list(channels.values())

    if d['DoStage'] == 'TRUE':
        n_sites = d['NStagePositions']
    else:
        n_sites = 1

    return channel_names, n_sites, d['NZSteps'], d['NTimePoints']

def extract_metadata_cv7k(img_files):
    channel_names = np.unique(
        [re.search("C[0-9]{2}(?=.tif)",
                   str(fyle)).group(0) for fyle in img_files])
    n_sites = len(np.unique(
        [re.search("F[0-9]{3}(?=L)",
                   str(fyle)).group(0) for fyle in img_files]))
    n_slices = len(np.unique(
        [re.search("Z[0-9]{2,}(?=C)",
                   str(fyle)).group(0) for fyle in img_files]))
    n_tp = len(np.unique(
        [re.search("T[0-9]{4}(?=F)",
                   str(fyle)).group(0) for fyle in img_files]))
    wells = np.unique([re.search("[a-zA-Z][0-9]{2}(?=_)",
                       str(fyle)).group(0) for fyle in img_files])

    return channel_names, n_sites, n_slices, n_tp, wells


def main():
    # load settings
    settings_path = Path(sys.argv[1])
    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)

    # get all paths from settings
    img_path = Path(settings['paths']['img_path'])
    output_path = Path(settings['paths']['zarr_path'])
    mag = settings['magnification']
    file_extension = settings['file_extension']
    microscope = settings['microscope']
    zspacing_um = settings['zspacing_um']

    if microscope == 'cv7k':
        img_files = list(img_path.glob(f'*.{file_extension}'))
        channel_names, n_sites, n_slices, n_tp = extract_metadata_cv7k(img_files)

    if microscope == 'visiscope':
        file = list(img_path.glob("*.nd"))[0]
        channel_names, n_sites, n_slices, n_tp = extract_metadata_visiscope(file)

    img_dims = IMG_DIMS[microscope]
    chunk = (1, n_slices, *img_dims)

    for site in range(1, n_sites+1):
        # Open the experiment zarr file
        if output_path.joinpath(f'site_{site:04}.zarr').exists():
            z = zarr.open(output_path.joinpath(f'site_{site:04}.zarr'), mode='a')
            print('opening in append mode')
        else:
            z = zarr.open(output_path.joinpath(f'site_{site:04}.zarr'), mode='w')
            print('making new file')
        for channel in channel_names:
            for level in range(0, N_LEVELS):
                if not hasattr(z, f'intensity_images/{channel}/level_{level}'):
                    d = z.create_dataset(
                        f'intensity_images/{channel}/level_{level:02d}',
                        shape=[n_tp, n_slices,
                               img_dims[0]/2**level,
                               img_dims[1]/2**level],
                        chunks=chunk,
                        dtype='uint16')

                    d.attrs["element_size_um"] = (zspacing_um,
                                                  PIXEL_SIZE_UM / mag * 2 ** level,
                                                  PIXEL_SIZE_UM / mag * 2 ** level)


if __name__ == "__main__":
   main()