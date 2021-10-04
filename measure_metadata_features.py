import h5py
from pathlib import Path
import pandas as pd
import sys
from lca.measure import measure_metadata, measure_morphology, measure_intensity, measure_blobs, measure_tracks
from lca.utils import measure_assignment
import numpy as np

# set paths
path_to_images = Path(r"Z:\20210930_dummy\hdf5")
path_to_features = Path(r"Z:\20210930_dummy\features")
path_to_images = Path(r'/data/active/atschan/20210930_dummy/hdf5/')
path_to_features = Path(r'/data/active/atschan/20210930_dummy/features/')

# define settings
settings = {'nuclei': {'measure_morphology' : True,
                       'measure_intensity' : ['sdcGFP', 'sdcDAPIxmRFPm'],
                       'measure_blobs' : ['sdcGFP'],
                       'measure_tracks' : True,
                       'assigned_objects' : [],
                       'aggregate' : False},
            'cells': {'measure_morphology' : True,
                      'measure_intensity' : ['sdcGFP', 'sdcDAPIxmRFPm'],
                      'measure_blobs' : [],
                      'measure_tracks' : False,
                      'assigned_objects' : ['nuclei', 'cytoplasm'],
                      'aggregate' : False},
            'cytoplasm': {'measure_morphology' : True,
                                 'measure_intensity' : ['sdcGFP', 'sdcDAPIxmRFPm'],
                                 'measure_blobs' : ['sdcDAPIxmRFPm'],
                                 'measure_tracks' : False,
                                 'assigned_objects' : [],
                                 'aggregate' : False}}

# define the site this job should process
site = int(sys.argv[1])

# load hdf5 file of site
file = h5py.File(path_to_images.joinpath('site_%04d.hdf5' % site), "a")


for object_id, object in enumerate(settings.keys()):

    label_images = file['label_images/%s' % object][:]
    object_settings = settings[object]

    # get metadata
    md = measure_metadata(label_images)
    fv = pd.DataFrame()

    # get morphology features
    if object_settings['measure_morphology']:
        morphology_features = measure_morphology(label_images)
        fv = fv.append(morphology_features)

    # get intensity features
    if object_settings['measure_intensity']:
        intensity_features = pd.DataFrame()

        for channel in object_settings['measure_intensity']:

            intensity_images = file['intensity_images'][channel][:]

            c_intensity_features = measure_intensity(label_images, intensity_images)
            c_intensity_features.columns = [t + '_%s' % channel for t in list(c_intensity_features.keys())]

            intensity_features = pd.concat([intensity_features, c_intensity_features], axis=1)

        # combine features to single dataframe
        fv = pd.concat([fv, intensity_features], axis=1)

    # add a unique identifier for every object, currently allows for 1000000 occurences per object per site
    unique_object_id = np.arange(0, len(md))+((site-1)*len(settings.keys())*1000000)+1000000*object_id

    md['unique_object_id'] = unique_object_id
    fv['unique_object_id'] = unique_object_id

    md = md.set_index('unique_object_id')
    fv = fv.set_index('unique_object_id')

    # get blob measurements
    if object_settings['measure_blobs']:
        for channel in object_settings['measure_blobs']:
            intensity_images = file['intensity_images'][channel][:]

            blobs = measure_blobs(label_images, intensity_images)
            blobs.to_csv(path_to_features.joinpath('blobs_%s_%s' % (object, channel)))
            blob_count = blobs.groupby(['label', 'timepoint']).size()
            fv['blob_count_%s' % channel]  = md.reset_index().set_index(['label', 'timepoint']).join(
                pd.DataFrame(blob_count, columns=['blob_count'], dtype='int')).set_index(
                'unique_object_id')['blob_count']


    # if object needs to be tracked across time, add track id
    if object_settings['measure_tracks'] == True:

        # track the objects in the dataframe and add the tracks to the metadata
        tracks = measure_tracks(md)
        md = md.join(tracks.set_index('unique_object_id'))

    # save metadata and feature values for this site
    md.to_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
    fv.to_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


# assignment between objects and aggregation if needed

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