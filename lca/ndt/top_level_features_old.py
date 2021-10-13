import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from lca.ndt.measure import measure_metadata_2DT, measure_morphology_2DT, measure_intensity_2DT, measure_blobs_2DT, measure_tracks_2DT
from lca.ndt.util import measure_assignment_2DT
import warnings

def extract_metadata_features(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, object in enumerate(settings['objects'].keys()):

        label_images = file['label_images/%s' % object][:]
        object_settings = settings['objects'][object]

        # get metadata
        md = measure_metadata_2DT(label_images)

        # define a unique identifier for every object, currently allows for 1000000 occurences per object per site
        unique_object_id = np.arange(0, len(md)) + ((site - 1) * len(settings.keys()) * 1000000) + 1000000 * object_id

        md['unique_object_id'] = unique_object_id

        # get morphology features
        if object_settings['measure_morphology']:
            morphology_features = measure_morphology_2DT(label_images)
            morphology_features['unique_object_id'] = unique_object_id

        # get intensity features
        if object_settings['measure_intensity']:
            intensity_features = []

            for channel in object_settings['measure_intensity']:
                intensity_images = file['intensity_images'][channel][:]

                c_intensity_features = measure_intensity_2DT(label_images, intensity_images)
                c_intensity_features['unique_object_id'] = unique_object_id
                intensity_features.append(c_intensity_features)

            suffixes = ['_%s' % channel for channel in object_settings['measure_intensity']]
            intensity_features = reduce(lambda df1, df2:
                                        pd.merge(df1, df2, on=('unique_object_id', 'label'),
                                                 suffixes=suffixes), intensity_features)

            # combine features to single dataframe
            fv = pd.merge(morphology_features, intensity_features, on=('unique_object_id', 'label'))

        md = md.set_index('unique_object_id')
        fv = fv.set_index('unique_object_id')

        # get blob measurements
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
                fv['blob_count_%s' % channel] = md.reset_index().set_index(['label', 'timepoint']).join(
                    pd.DataFrame(blob_count, columns=['blob_count'], dtype='int')).set_index(
                    'unique_object_id')['blob_count']

        # if object needs to be tracked across time, add track id
        if object_settings['measure_tracks'] == True:
            # track the objects in the dataframe and add the tracks to the metadata
            tracks = measure_tracks_2DT(md)
            md = md.join(tracks.set_index('unique_object_id'))

        # save metadata and feature values for this site
        md.to_csv(feature_path.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
        fv.to_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))


def assign_and_aggregate(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, object in enumerate(settings['objects'].keys()):

        if settings['objects'][object]['assigned_objects']:

            md = pd.read_csv(feature_path.joinpath('site_%04d_%s_metadata.csv' % (site, object)))
            fv = pd.read_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
            fv = fv.set_index('unique_object_id')

            label_images = file['label_images/%s' % object][:]

            for assigned_object in settings['objects'][object]['assigned_objects']:

                md_assignment = pd.read_csv(
                    feature_path.joinpath('site_%04d_%s_metadata.csv' % (site, assigned_object)))
                fv_assignment = pd.read_csv(
                    feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, assigned_object)))

                assigned_label_images = file['label_images/%s' % assigned_object][:]

                md_assignment = measure_assignment_2DT(label_images, assigned_label_images, md, md_assignment)
                fv_assignment['unique_object_id'] = md_assignment['unique_object_id']

                array1 = np.array(md_assignment.reset_index().unique_object_id)
                array2 = np.array(md.reset_index().unique_object_id)

                if np.array_equal(array1, array2):
                    fv_assignment = fv_assignment.set_index('unique_object_id')

                else:
                    warnings.warn(' %s objects do not have a 1-to-1 relationship to %s . They will be aggregated.' % (
                    assigned_object, object))
                    counts = fv_assignment.groupby('unique_object_id').size()
                    fv_assignment = fv_assignment.groupby('unique_object_id').mean()
                    fv_assignment['count'] = counts

                    '''
                    aggregating the md will have some negative effects - for example the track id will be a mean measurement as well...
                    but so far I can't think of a better way to deal with this if two tracked objects are present in '''

                    md_assignment = md_assignment.groupby('unique_object_id').mean()
                    md_assignment['count'] = counts

                # rename the columns of dataframe to be assigned to another and join the dfs
                md_assignment = md_assignment.reset_index().set_index(['unique_object_id', 'timepoint'])
                md = md.set_index(['unique_object_id', 'timepoint'])

                fv_assignment.columns = [t + '_%s' % assigned_object for t in list(fv_assignment.keys())]
                md_assignment.columns = [t + '_%s' % assigned_object for t in list(md_assignment.keys())]

                fv = fv.join(fv_assignment)
                md = md.join(md_assignment)

                md = md.reset_index()

            # reset index and save
            fv = fv.reset_index().set_index('unique_object_id')
            md = md.reset_index().set_index('unique_object_id')

            fv.to_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, object)))
            md.to_csv(feature_path.joinpath('site_%04d_%s_metadata.csv' % (site, object)))

        else:
            pass

def main(file, settings):
    site = int(file.filename.replace('.hdf5', '').split('site_')[1])
    extract_metadata_features(file, settings, site)
    assign_and_aggregate(file, settings, site)


if __name__ == "__main__":
    main()