import zarr
import yaml
import pandas as pd
from pathlib import Path
import napari


# load settings
settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20220414_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)


feature_path = Path(settings['paths']['feature_path'])

fv = pd.read_csv(feature_path.joinpath('site_0003_nuclei_feature_values.csv'))

import btrack


from lca.ndt.measure_new import measure_tracks_2DT
import trackpy as tp


test = tp.link(fv, search_range=10, memory=5, pos_columns=['centroid-1', 'centroid-2'], t_column='timepoint')


to_track = fv[['timepoint', 'centroid-1', 'centroid-2', 'label']]

to_track['centroid-0'] = to_track['centroid-1']
to_track['centroid-1'] = to_track['centroid-2']

test = measure_tracks_2DT(to_track[['timepoint', 'centroid-0', 'centroid-1', 'label']],
                          max_distance=10, time_window=3, max_split_distance=0, max_gap_closing_distance=15, allow_splitting=False, allow_merging=False)

viewer = napari.Viewer()

viewer.add_tracks(test[['track_id', 'timepoint', 'centroid-0', 'centroid-1']])
viewer.add_tracks(test[['particle', 'timepoint', 'centroid-0', 'centroid-1', 'centroid-2']])
# define which level should be used for segmentation
level = 'level_03'

# define which channel should be used for segmentation
channel = settings['cellpose']['nuclei']['channel']

# get all paths from settings
zarr_path = Path(settings['paths']['zarr_path'])

zarr_files = zarr_path.glob('*.zarr')
zarr_files = [fyle for fyle in zarr_files]

fyle = zarr_files[0]

z = zarr.open(fyle, mode='a')

img = z['intensity_images'][channel][level][timepoint, :, 0:100, 0:100]

#img = ndi.median_filter(img, 3)
filter_sigma=1
image = sitk.GetImageFromArray(img)
image.SetSpacing((1.0, 1.0, 1.25))
gaussianfilter = sitk.SmoothingRecursiveGaussianImageFilter()
gaussianfilter.SetSigma(filter_sigma)
img = sitk.GetArrayFromImage(gaussianfilter.Execute(image))

filter = sitk.MedianImageFilter()
filter.SetRadius(3)

viewer = napari.Viewer()
viewer.add_image(img)
