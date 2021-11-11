import numpy as np
import pandas as pd
from pathlib import Path
from lca.ndt.measure import measure_morphology_2DT, measure_intensity_2DT, measure_blobs_2DT, measure_tracks_2DT
from lca.ndt.util import measure_assignment_2DT
from lca.util import measure_border_cells
import warnings

def extract_features(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, object in enumerate(settings['objects'].keys()):

        fv = []
        label_images = file['label_images/%s' % object][:]
        object_settings = settings['objects'][object]

        # get morphology features
        if object_settings['measure_morphology']:
            morphology_features = measure_morphology_2DT(label_images)
            fv.append(morphology_features)

        # get intensity features
        if object_settings['measure_intensity']:

            intensity_features = []

            for channel in object_settings['measure_intensity']:
                intensity_images = file['intensity_images'][channel][:]

                c_intensity_features = measure_intensity_2DT(label_images, intensity_images)
                c_intensity_features = c_intensity_features.set_index(['label', 'timepoint'])
                c_intensity_features.columns = ['%s_%s' % (key, channel) for key in c_intensity_features.columns]
                c_intensity_features = c_intensity_features.reset_index()

                intensity_features.append(c_intensity_features)

            intensity_features = [feature.set_index(['label', 'timepoint']) for feature in intensity_features]
            fv.append(pd.concat(intensity_features, axis=1).reset_index())

        fv = [feature.set_index(['label', 'timepoint']) for feature in fv]
        fv = pd.concat(fv, axis=1).reset_index()

        # get blob fv
        if object_settings['measure_blobs']:
            for channel in object_settings['measure_blobs']['channels']:
                intensity_images = file['intensity_images'][channel][:]

                blobs = measure_blobs_2DT(intensity_images, label_images,
                                          min_sigma=object_settings['measure_blobs']['settings']['min_sigma'],
                                          max_sigma=object_settings['measure_blobs']['settings']['max_sigma'],
                                          num_sigma=object_settings['measure_blobs']['settings']['num_sigma'],
                                          threshold=object_settings['measure_blobs']['settings']['threshold'],
                                          overlap=object_settings['measure_blobs']['settings']['overlap'],
                                          exclude_border=object_settings['measure_blobs']['settings']['exclude_border'])

                blobs.to_csv(feature_path.joinpath('site_%04d_blobs_%s_%s.csv' % (site, object, channel)))
                blob_count = blobs.groupby(['label', 'timepoint']).size()

                fv = fv.merge(blob_count.rename('blob_count_%s' % channel), on=['label', 'timepoint'], how='outer')
                fv['blob_count_%s' % channel].fillna(0, inplace=True)
                

        # if object needs to be tracked across time, add track id
        if object_settings['measure_tracks']:
            # track the objects in the dataframe
            fv = measure_tracks_2DT(fv,
                                    max_distance=object_settings['measure_tracks']['max_distance'],
                                    time_window=object_settings['measure_tracks']['time_window'],
                                    max_split_distance=object_settings['measure_tracks']['max_split_distance'],
                                    max_gap_closing_distance=object_settings['measure_tracks']['max_gap_closing_distance'],
                                    allow_splitting=object_settings['measure_tracks']['allow_splitting'],
                                    allow_merging=object_settings['measure_tracks']['allow_merging'])

        # add unique_object_id
        unique_object_id = np.arange(0, len(fv)) + ((site - 1) * len(settings.keys()) * 1000000) + 1000000 * object_id
        fv['unique_object_id'] = unique_object_id

        # measure if object touches the border
        fv['is_border'] = measure_border_cells(fv)

        # save  feature values for this site
        fv = fv.set_index(['unique_object_id', 'timepoint'])
        fv.to_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


def assign_and_aggregate2(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, object in enumerate(settings['objects'].keys()):

        if settings['objects'][object]['assigned_objects']:

            fv = pd.read_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
            label_images = file['label_images/%s' % object][:]

            for assigned_object in settings['objects'][object]['assigned_objects']:

                fv_assignment = pd.read_csv(
                    feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, assigned_object)))

                assigned_label_images = file['label_images/%s' % assigned_object][:]

                parent_lbl_df = get_parent_labels_2DT(assigned_label_images, label_images)
                fv_assignment = pd.merge(fv_assignment, parent_lbl_df,
                                         left_on=['label', 'timepoint'],
                                         right_on=['label', 'timepoint'])

                # check if assigned objects have 1-to-1 relationship to parent object
                array1 = np.array(fv_assignment.reset_index().unique_object_id)
                array2 = np.array(fv.reset_index().unique_object_id)

                if np.array_equal(array1, array2):
                    fv_assignment = fv_assignment.set_index('unique_object_id')

                else:
                    warnings.warn(' %s objects do not have a 1-to-1 relationship to %s . They will be aggregated.' % (
                    assigned_object, object))
                    counts = fv_assignment.groupby('unique_object_id').size()
                    fv_assignment = fv_assignment.groupby('unique_object_id').mean()
                    fv_assignment['count'] = counts

                    '''
                    aggregating the fv will have some negative effects - for example the track id will be a 
                    mean measurement as well... but so far I can't think of a better way to deal 
                    with this if two tracked objects are present in 
                    '''

                '''
                Design decision to be made: Do we want to keep cells without a nucleus in the dataframe?
                If yes -> how='outer' If no, leave default value. Keeping them for now.
                '''
                # merge the dfs while renaming according to the assigned object

                fv_assignment = fv_assignment.set_index(['timepoint']).add_suffix('_%s' % assigned_object)
                fv = fv.merge(fv_assignment, left_on=['label', 'timepoint'],
                              right_on=['parent_label_%s' % assigned_object, 'timepoint'], how='outer')


            # reset index and save
            fv = fv.set_index(['unique_object_id', 'timepoint'])
            fv.to_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


        else:
            pass


def assign_and_aggregate(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, object in enumerate(settings['objects'].keys()):

        if settings['objects'][object]['assigned_objects']:

            fv = pd.read_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
            label_images = file['label_images/%s' % object][:]

            for assigned_object in settings['objects'][object]['assigned_objects']:

                fv_assignment = pd.read_csv(
                    feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, assigned_object)))

                assigned_label_images = file['label_images/%s' % assigned_object][:]

                fv_assignment = measure_assignment_2DT(label_images, assigned_label_images, fv, fv_assignment)

                # check if assigned objects have 1-to-1 relationship to parent object
                array1 = np.array(fv_assignment.reset_index().unique_object_id)
                array2 = np.array(fv.reset_index().unique_object_id)

                if np.array_equal(array1, array2):
                    fv_assignment = fv_assignment.set_index('unique_object_id')

                else:
                    warnings.warn(' %s objects do not have a 1-to-1 relationship to %s . They will be aggregated.' % (
                    assigned_object, object))
                    counts = fv_assignment.groupby('unique_object_id').size()
                    fv_assignment = fv_assignment.groupby('unique_object_id').mean()
                    fv_assignment['count'] = counts

                    '''
                    aggregating the fv will have some negative effects - for example the track id will be a 
                    mean measurement as well... but so far I can't think of a better way to deal 
                    with this if two tracked objects are present in 
                    '''

                '''
                Design decision to be made: Do we want to keep cells without a nucleus in the dataframe?
                If yes -> how='outer' If no, leave default value. Keeping them for now.
                '''
                # merge the dfs while renaming according to the assigned object

                fv_assignment = fv_assignment.reset_index().set_index(['unique_object_id', 'timepoint'])
                fv_assignment.columns = ['%s_%s' % (key, assigned_object) for key in fv_assignment.columns]
                fv_assignment = fv_assignment.reset_index()
                fv = fv.merge(fv_assignment, on=['unique_object_id', 'timepoint'], how='outer')


            # reset index and save
            fv = fv.set_index(['unique_object_id', 'timepoint'])
            fv.to_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


        else:
            pass

def main(file, settings):
    site = int(file.filename.replace('.hdf5', '').split('site_')[1])
    extract_features(file, settings, site)
    assign_and_aggregate(file, settings, site)


if __name__ == "__main__":
    main()