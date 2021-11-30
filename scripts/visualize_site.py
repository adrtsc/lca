import napari
from pathlib import Path
import h5py
import yaml
import pandas as pd


def visualize_site(settings, site, intensities=True, labels=True, boundaries=True, tracks=True, blobs=True):

    hdf5_path = Path(settings['paths']['hdf5_path'])
    feature_path = Path(settings['paths']['feature_path'])

    # load hdf5 file of site
    viewer = napari.Viewer()

    with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "r") as file:
        if intensities:
            # add intensity images
            for channel in file['intensity_images'].keys():
                intensity_image = file['intensity_images'][channel][:]
                viewer.add_image(intensity_image, name=channel, blending='additive',
                                 colormap=settings['channel_colors'][channel])
        if labels:
            # add label images
            for obj in file['label_images'].keys():
                label_image = file['label_images'][obj][:].astype('uint16')
                viewer.add_labels(label_image, name=obj, blending='additive', visible=False)
        if boundaries:
            # add boundary images
            for obj in file['boundary_images'].keys():
                boundary_image = file['boundary_images'][obj][:].astype('uint16')
                viewer.add_image(boundary_image, name=obj, blending='additive', visible=False)

        if tracks:
            # add tracks
            for obj in file['label_images'].keys():
                try:
                    fv = pd.read_csv(feature_path.joinpath('site_%04d_%s_feature_values.csv' % (site, obj)))
                    if hasattr(fv, 'track_id'):
                        tracks = fv[['track_id', 'timepoint', 'centroid-0', 'centroid-1']].astype('uint16')
                        viewer.add_tracks(tracks, name='tracks_%s' % obj, visible=False)
                except:
                    pass
    if blobs:
        # add blobs
        blob_files = feature_path.glob('*.csv')
        blob_files = [blob_file for blob_file in blob_files if 'site_%04d_blobs_' % site in str(blob_file)]


        for blob_file in blob_files:
            blobs = pd.read_csv(blob_file)
            viewer.add_points(blobs[['timepoint', 'centroid-0', 'centroid-1']], name=str(blob_file).replace('.csv', '').split('site_%04d_' % site)[1],
                              face_color='transparent',
                              edge_color='white',
                              size=blobs['size'],
                              visible=False)




# load settings
with open('scripts/settings/20211126_Sec16_live_settings.yml', 'r') as stream:
    settings = yaml.safe_load(stream)

visualize_site(settings, 1)