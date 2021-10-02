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

# define settings

settings = {'nuclei': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'], 'assigned_object':'cells', 'aggregate':False},
            'cells': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'], 'assigned_object':'cells', 'aggregate':False},
            'nuclear_speckles': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'], 'assigned_object':'cells', 'aggregate':True}}

for object_id, object in enumerate(settings.keys()):

    label_images = file['label_images/%s' % object][:]
    #assignment_label_images = file['label_images/%s' % settings[object]['assigned_object']][:]
    intensity_images =[file['intensity_images'][channel][:] for channel in settings[object]['channels']]

    morphology_features = measure_morphology(label_images)
    intensity_features = measure_intensity(label_images, intensity_images, settings[object]['channels'])

    # combine features to single dataframe
    feature_values = pd.concat([morphology_features, intensity_features], axis=1)

    # add a unique identifier for every object, currently allows for 10000 occurences per object per site
    feature_values['unique_id'] = np.arange(0, len(feature_values))
    feature_values['unique_id'] = feature_values['unique_id']+(site*len(settings.keys())*10000)+10000*object_id
    feature_values = feature_values.set_index('unique_id')


    # save feature values for this site
    feature_values.to_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


# assignment between objects
'''
two things to do: if there is a 1:1 relationship, one can just concatenate the dataframes. That's simple
if there is a many-to-one relationship, one has to aggregate the measurements
first one would have to figure out which objects are assigned together - for simplicity let's assume we only
want to assign everything to cells
'''

path_to_assigned_features = Path(r"Z:\20210930_dummy\features\assigned")

for object_id, object in enumerate(settings.keys()):

    if object != settings[object]['assigned_object']:

        assigned_object = settings[object]['assigned_object']
        md = pd.read_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
        fv = pd.read_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
        md_assignment = pd.read_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, assigned_object)))
        fv_assignment = pd.read_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, assigned_object)))

        label_images = file['label_images/%s' % object][:]
        assigned_label_images = file['label_images/%s' % settings[object]['assigned_object']][:]

        assigned_md = measure_assignment(label_images, assigned_label_images, md, md_assignment)
        fv['unique_id'] = assigned_md['unique_id']

        if settings[object]['aggregate'] == True:
            fv = fv.groupby('unique_id').mean()

        else:
            fv = fv.set_index('unique_id')

        fv_assignment = fv_assignment.set_index('unique_id')

        # rename the columns of dataframe to be assigned to another
        fv.columns = [ t + '_%s' % object for t in list(fv.keys())]

        result = fv_assignment.join(fv)
        result = result.drop('label_%s' % object, axis='columns')

        result.to_csv(path_to_features.joinpath('site_%04d_%s_feature_values.csv' % (site, assigned_object)))







'''

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


'''