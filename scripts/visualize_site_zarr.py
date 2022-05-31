import napari
from pathlib import Path
import zarr
import yaml
import pandas as pd
import numpy as np


def visualize_site(settings,
                   site,
                   start_timepoint,
                   end_timepoint,
                   level,
                   intensities=True,
                   labels=True,
                   boundaries=True,
                   tracks=True,
                   blobs=True):

    zarr_path = Path(settings['paths']['zarr_path'])
    feature_path = Path(settings['paths']['feature_path'])

    # load hdf5 file of site
    viewer = napari.Viewer()
    
    file = zarr.open(zarr_path.joinpath(f'site_{site:04d}.zarr'), "r")


    if intensities:
        # add intensity images
        for channel in file['intensity_images'].keys():
            intensity_image = file['intensity_images'][channel][f'level_{level:02d}'][start_timepoint:end_timepoint, :, :, :]
            scaling = file['intensity_images'][channel][f'level_{level:02d}'].attrs['element_size_um']
            viewer.add_image(intensity_image, name=channel, blending='additive',
                             colormap=settings['channel_colors'][channel],
                             scale=scaling)
    if labels:
        # add label images
        for obj in file['label_images'].keys():
            label_image = file['label_images'][obj][f'level_{level:02d}'][start_timepoint:end_timepoint, :, :, :].astype('uint16')
            scaling = file['intensity_images'][channel][f'level_{level:02d}'].attrs['element_size_um']
            viewer.add_labels(label_image, name=obj, blending='additive', visible=False,
                             scale=scaling)
    if boundaries:
        # add boundary images
        for obj in file['boundary_images'].keys():
            boundary_image = file['boundary_images'][obj][start_timepoint:end_timepoint, :, :, :].astype('uint16')
            viewer.add_image(boundary_image, name=obj, blending='additive', visible=False,
                             scale=(settings['scaling'], 1, 1))

    if tracks:
        # add tracks
        for obj in file['label_images'].keys():
            try:
                fv = pd.read_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, obj)))
                fv['centroid-1_scaled'] = fv['centroid-1'] / 2 ** level
                fv['centroid-2_scaled'] = fv['centroid-2'] / 2 ** level
                if hasattr(fv, 'track_id'):
                    tracks = fv[['track_id', 'timepoint', 'centroid-0', 'centroid-1', 'centroid-2']].astype('uint16')
                    viewer.add_tracks(tracks, name='tracks_%s' % obj, visible=False)
            except:
                pass

    if blobs:
        # add blobs
        blob_files = feature_path.glob('distance_measurements/*.csv')
        blob_files = [blob_file for blob_file in blob_files if 'site_%04d' % site in str(blob_file)]


        for blob_file in blob_files:
            blobs = pd.read_csv(blob_file)
            filter = blobs['timepoint'].isin(np.arange(start_timepoint, end_timepoint))
            blobs = blobs[filter]
            blobs['centroid-1_scaled'] = blobs['centroid-1_blobs'] / 2**level
            blobs['centroid-2_scaled'] = blobs['centroid-2_blobs'] / 2**level
            blobs['centroid-0_scaled'] = blobs['centroid-0_blobs'] / 9.25
            scaling_blobs = scaling.copy()
            scaling_blobs.insert(0, 1)
            viewer.add_points(blobs[['timepoint', 'centroid-0_scaled', 'centroid-1_scaled', 'centroid-2_scaled']], name=str(blob_file).replace('.csv', '').split('site_%04d' % site)[1],
                              face_color='transparent',
                              edge_color='white',
                              size=blobs['size_blobs']/2**level,
                              visible=False,
                              scale=scaling_blobs)

    return viewer


# load settings
with open('scripts/settings/20220414_settings.yml', 'r') as stream:
    settings = yaml.safe_load(stream)

viewer = visualize_site(settings, 2, start_timepoint=0, end_timepoint=79, level=3, boundaries=False, blobs=False, tracks=True, labels=False)
