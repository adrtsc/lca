import zarr
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
#import napari
#from microfilm import colorify
import sys
from skimage import exposure
from skimage import io

def get_images(sample_df, zarr_path, output_path, class_label, channel, level):

    for track_id in sample_df['unique_track_id'].unique():

        track = sample_df[sample_df['unique_track_id'] == track_id]
        output_path.joinpath(f'TSS_clustering/class_{class_label + 1}/track_{track_id}').mkdir(parents=True, exist_ok=True)

        print(f'track_{track_id}')

        for timepoint in np.array([0, 9, 19, 20, 29, 39, 49, 59]):
            print(timepoint)
            track_tp = track[track['timepoint'] == timepoint]
            site = track_tp.site.unique()[0]
            z = zarr.open(zarr_path.joinpath(f'site_{site + 1:04d}.zarr'),
                          mode='r')

            x_low = int(track_tp['bbox-2'])
            x_high = int(track_tp['bbox-5'])
            y_low = int(track_tp['bbox-1'])
            y_high = int(track_tp['bbox-4'])

            label = int(track_tp['label'])

            lbl = z[f'label_images/nuclei/level_{level:02d}'][timepoint, :,
                  :, :]
            lbl = lbl[:, y_low: y_high, x_low: x_high]
            lbl[lbl != label] = 0

            img = z[f'intensity_images/{channel}/level_{level:02d}'][
                  timepoint, :, :, :]

            img = img[:, y_low: y_high, x_low: x_high]
            img[lbl != label] = 0
            img = img.max(axis=0)

            img = exposure.rescale_intensity(img, in_range=(115, 350))
            #cmap = microfilm.colorify.cmaps_def("magma", num_colors=2**16, flip_map=False)
            #img = colorify.colorify_by_cmap(image=img, cmap=cmap)[0]

            io.imsave(output_path.joinpath(f'TSS_clustering/class_{class_label+1}/track_{track_id}/{class_label+1}_{track_id}_timepoint_{timepoint}.png'), img)




def main():


    class_label = int(sys.argv[1])
    level = 0
    channel = 'sdc-RFP-605-52'

    #settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20220224_settings.yml")
    settings_path = Path(
        r"/data/homes/atschan/PhD/Code/Python/lca/scripts/settings/20220224_cluster_settings.yml")
    #output_path = Path(r"Z:\20220224_hiPSC_MS2\classification_examples")
    output_path = Path(r"/data/active/atschan/20220224_hiPSC_MS2/classification_examples")

    fv = pd.read_csv(r"/data/active/atschan/20220224_hiPSC_MS2/features/20220224_fv_preprocessed_step2_clustered.csv")
    #fv = pd.read_csv(
    #    r"Z:\20220224_hiPSC_MS2\features\20220224_fv_preprocessed_step2_clustered.csv")

    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)

    zarr_path = Path(settings['paths']['zarr_path'])
    scale_corr = 0.8667

    # take 5 sample tracks of each cluster
    fv = fv.loc[fv['mock'] == False]
    sample = fv.loc[fv['TSS_class_label'] == class_label].sample(10)['unique_track_id']
    sample_df = fv[np.isin(fv['unique_track_id'], sample)]

    # correct coordinates for the current level

    sample_df['centroid-1'] = sample_df['centroid-1']*2**(3-level)
    sample_df['centroid-2'] = sample_df['centroid-2']*2**(3-level)

    sample_df['bbox-2'] = sample_df['bbox-2']*2**(3-level)
    sample_df['bbox-5'] = sample_df['bbox-5']*2**(3-level)
    sample_df['bbox-1'] = sample_df['bbox-1']*2**(3-level)
    sample_df['bbox-4'] = sample_df['bbox-4']*2**(3-level)


    sample_df['centroid-1'] = sample_df['centroid-1']/scale_corr
    sample_df['centroid-2'] = sample_df['centroid-2']/scale_corr


    get_images(sample_df, zarr_path, output_path, class_label, channel, level)


if __name__ == "__main__":
    main()
