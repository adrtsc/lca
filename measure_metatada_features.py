import h5py
from pathlib import Path
import pandas as pd
import sys
from lca.LapTracker import LapTracker
from lca.features import Metadata, Morphology, Intensity
from lca.utils import measure_assignment
import numpy as np

# set paths
path_to_images = Path(r"Z:\20210930_dummy\hdf5")
path_to_features = Path(r"Z:\20210930_dummy\features")

# define settings
settings = {'nuclei': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'],
                       'assigned_objects':[],
                       'aggregate':False,
                       'track':True},
            'cells': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'],
                      'assigned_objects':['nuclei', 'nuclear_speckles'],
                      'aggregate':False,
                      'track':False},
            'nuclear_speckles': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'],
                                 'assigned_objects':[],
                                 'aggregate':True,
                                 'track':False}}

# define the site this job should process
site = int(sys.argv[1])

# load hdf5 file of site
file = h5py.File(path_to_images.joinpath('site_%04d.hdf5' % site), "a")


for object_id, object in enumerate(settings.keys()):

    label_images = file['label_images/%s' % object][:]

    # get metadata
    md = Metadata(label_images).extract()

    # get morphology features
    morphology_features = Morphology(label_images).extract()

    # get intensity features
    intensity_features = pd.DataFrame()

    for channel in settings[object]['channels']:

        intensity_images = file['intensity_images'][channel][:]

        c_intensity_features = Intensity(label_images, intensity_images).extract()
        c_intensity_features.columns = [t + '_%s' % channel for t in list(c_intensity_features.keys())]

        intensity_features = pd.concat([intensity_features, c_intensity_features], axis=1)

    # combine features to single dataframe
    fv = pd.concat([morphology_features, intensity_features], axis=1)

    # add a unique identifier for every object, currently allows for 10000 occurences per object per site
    unique_object_id = np.arange(0, len(md))+((site-1)*len(settings.keys())*100000)+100000*object_id

    md['unique_object_id'] = unique_object_id
    fv['unique_object_id'] = unique_object_id

    md = md.set_index('unique_object_id')
    fv = fv.set_index('unique_object_id')

    # if object needs to be tracked across time, add track id

    if settings[object]['track'] == True:

        # track the objects in the dataframe

        tracker = LapTracker(max_distance=30, time_window=3, max_split_distance=100,
                             max_gap_closing_distance=100)

        columns = ['centroid-0', 'centroid-1', 'timepoint', 'label']
        md['track_id'] = tracker.track_df(md, identifiers=columns)

    # save metadata and feature values for this site
    md.to_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
    fv.to_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


# assignment between objects
'''
two things to do: if there is a 1:1 relationship, one can just concatenate the dataframes. That's simple
if there is a many-to-one relationship, one has to aggregate the measurements (at the moment I only do mean aggregation)
'''

for object_id, object in enumerate(settings.keys()):

    if settings[object]['assigned_objects']:

        md = pd.read_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
        fv = pd.read_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
        fv = fv.set_index('unique_object_id')

        label_images = file['label_images/%s' % object][:]

            for assigned_object in settings[object]['assigned_objects']:

                md_assignment = pd.read_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, assigned_object)))
                fv_assignment = pd.read_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, assigned_object)))

                assigned_label_images = file['label_images/%s' % assigned_object][:]

                assigned_md = measure_assignment(assigned_label_images, label_images, md_assignment, md)
                fv_assignment['unique_object_id'] = assigned_md['unique_object_id']

                if settings[assigned_object]['aggregate'] == True:
                    counts = fv_assignment.groupby('unique_object_id').size()
                    fv_assignment = fv_assignment.groupby('unique_object_id').mean()
                    fv_assignment['count'] = counts

                else:
                    fv_assignment = fv_assignment.set_index('unique_object_id')

                # rename the columns of dataframe to be assigned to another and join the dfs
                fv_assignment.columns = [ t + '_%s' % assigned_object for t in list(fv_assignment.keys())]

                fv = fv.join(fv_assignment)

                # check if assigned object was tracked and add track id if it was
                if hasattr(md_assignment, 'track_id'):
                    md['track_id_%s' % assigned_object] = md_assignment['track_id']

            fv.to_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
            md.to_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
    else:
        pass