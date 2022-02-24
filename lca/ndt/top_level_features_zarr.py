import numpy as np
import pandas as pd
from pathlib import Path
from lca.ndt.measure import (measure_morphology_3DT,
                             measure_intensity_3DT,
                             measure_blobs_3DT,
                             measure_tracks_2DT)
from lca.ndt.util import measure_assignment_2DT
from lca.util import measure_border_cells
import warnings

def extract_features(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, obj in enumerate(settings['objects'].keys()):

        fv = []
        lvl = settings['objects']['nuclei']['measure_morphology']
        label_images = file[f'label_images/{obj}/{lvl}'][:].astype('uint16')
        object_settings = settings['objects'][obj]

        # get morphology features

        spacing = file[f'label_images/{obj}/{lvl}'].attrs['element_size_um']

        if object_settings['measure_morphology']:
            morphology_features = measure_morphology_3DT(label_images,
                                                         spacing)
            fv.append(morphology_features)

        print(f'measured morphology features of {obj}')

        # get intensity features
        if object_settings['measure_intensity']:

            intensity_features = []

            for channel in object_settings['measure_intensity']:

                lvl = object_settings['measure_intensity'][channel]

                intensity_images = file[f'intensity_images/{channel}/{lvl}'][:]

                spacing = file[f'intensity_images/{channel}/{lvl}'].attrs['element_size_um']

                current_features = measure_intensity_3DT(label_images,
                                                         intensity_images,
                                                         spacing)
                current_features = current_features.set_index(['label',
                                                               'timepoint'])
                current_features.columns = [
                    f'{key}_{channel}' for key in current_features.columns]
                current_features = current_features.reset_index()

                intensity_features.append(current_features)

            intensity_features = [feature.set_index(['label', 'timepoint']) for
                feature in intensity_features]
            fv.append(pd.concat(intensity_features, axis=1).reset_index())

            print(f'measured intensity features of {obj}')

        fv = [feature.set_index(['label', 'timepoint']) for feature in fv]
        fv = pd.concat(fv, axis=1).reset_index()

        # get blob fv
        if object_settings['measure_blobs']:

            blob_settings = object_settings['measure_blobs']['settings']

            for channel in object_settings['measure_blobs']['channels']:
                intensity_images = file[f'intensity_images/{channel}/level_00'][:]
                label_images = file[f'label_images/{obj}/level_00'][:]

                blobs = measure_blobs_3DT(
                    intensity_images=intensity_images,
                    label_images=label_images,
                    **blob_settings)

                blobs.to_csv(
                    feature_path.joinpath(
                        f'site_{site:04}_blobs_{obj}_{channel}.csv'))
                blob_count = blobs.groupby(['label', 'timepoint']).size()

                fv = fv.merge(blob_count.rename(f'blob_count_{channel}'),
                              on=['label', 'timepoint'], how='outer')
                fv[f'blob_count_{channel}'].fillna(0, inplace=True)
                

        # if obj needs to be tracked across time, add track id
        if object_settings['measure_tracks']:

            track_settings = object_settings['measure_tracks']
            # track the objects in the dataframe
            fv = measure_tracks_2DT(df=fv, **track_settings)

            # add unique_track_id
            fv['unique_track_id'] = fv['track_id']+site*10000

        # add unique_object_id
        unique_object_id = (np.arange(0, len(fv)) +
                            ((site - 1) * len(settings.keys()) * 1000000) +
                            1000000 * object_id)
        fv['unique_object_id'] = unique_object_id

        # save  feature values for this site
        fv = fv.set_index(['unique_object_id', 'timepoint'])
        fv.to_csv(feature_path.joinpath(
            f'site_{site:04}_{obj}_feature_values.csv'))


def assign_and_aggregate(file, settings, site):

    feature_path = Path(settings['paths']['feature_path'])

    for object_id, obj in enumerate(settings['objects'].keys()):

        object_settings = settings['objects'][obj]

        if object_settings['assigned_objects']:

            fv = pd.read_csv(
                feature_path.joinpath(
                    f'site_{site:04}_{obj}_feature_values.csv'))

            label_images = file['label_images/%s' % obj][:].astype('uint16')

            for assigned_object in object_settings['assigned_objects']:

                fv_assignment = pd.read_csv(
                    feature_path.joinpath(
                        f'site_{site:04}_{assigned_object}_feature_values.csv'))

                assigned_label_images = file['label_images/{assigned_object}'][:]

                fv_assignment = measure_assignment_2DT(label_images,
                                                       assigned_label_images,
                                                       fv,
                                                       fv_assignment)

                # check 1-to-1 relationship
                array1 = np.array(fv_assignment.reset_index().unique_object_id)
                array2 = np.array(fv.reset_index().unique_object_id)

                if np.array_equal(array1, array2):
                    fv_assignment = fv_assignment.set_index('unique_object_id')

                else:
                    warnings.warn((f'{assigned_object} objects do not have'
                                   f'a 1-to-1 relationship to {obj}. '
                                   'They will be aggregated.'))
                    counts = fv_assignment.groupby('unique_object_id').size()
                    fv_assignment = fv_assignment.groupby(
                        'unique_object_id').mean()
                    fv_assignment['count'] = counts

                    '''
                    aggregating the fv will have some negative effects - 
                    for example the track id will be a mean measurement as well
                    but so far I can't think of a better way to deal with this
                    if two tracked objects are present in 
                    '''

                '''
                Design decision to be made: Do we want to keep cells without
                a nucleus in the dataframe? If yes -> how='outer' If no, leave
                default value. Keeping them for now.
                '''
                # merge the dfs while renaming according to the assigned obj

                fv_assignment = fv_assignment.reset_index().set_index(
                    ['unique_object_id', 'timepoint'])
                fv_assignment.columns = [f'{key}_{assigned_object}' for
                                         key in fv_assignment.columns]
                fv_assignment = fv_assignment.reset_index()
                fv = fv.merge(fv_assignment,
                              on=['unique_object_id', 'timepoint'],
                              how='outer')


            # reset index and save
            fv = fv.set_index(['unique_object_id', 'timepoint'])
            fv.to_csv(
                feature_path.joinpath(
                    'site_{site:04}_{obj}_feature_values.csv'))


        else:
            pass

def main(file, filename, settings):
    site = int(filename.replace('.zarr', '').split('site_')[1])
    extract_features(file, settings, site)
    assign_and_aggregate(file, settings, site)


if __name__ == "__main__":
    main()