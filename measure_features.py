import h5py
from pathlib import Path
import skimage.measure
import pandas as pd
import sys
from LapTracker.LapTracker import LapTracker
import napari

path_to_images = Path(r"Z:\20210930_dummy\hdf5")
path_to_features = Path(r"Z:\20210930_dummy\features")

# define the site this job should process
site = int(sys.argv[1])

# load hdf5 file of site
file = h5py.File(path_to_images.joinpath('site_%04d.hdf5' % site), "a")

# measure morphology features
morphology_features = measure_morphology(file)

# measure intensity features
intensity_features = measure_intensity(file)

# combine features to single dataframe
feature_values = pd.concat([morphology_features, intensity_features], axis=1)

# measure border cells
feature_values['border_cells'] = measure_border_cells(feature_values)

# track the objects in the dataframe

tracker = LapTracker(max_distance=30, time_window=3, max_split_distance=100,
                     max_gap_closing_distance=100)

columns = ['centroid-0_nuclei', 'centroid-1_nuclei', 'timepoint', 'label_nuclei']
tracker.track_df(feature_values, identifiers=columns)

feature_values = tracker.df

# adapt unique id for site
feature_values['unique_id'] = feature_values['unique_id'] + (site*10000)

# save feature values for this site

feature_values.to_csv(path_to_features.joinpath('site_%04d.csv' % site))


