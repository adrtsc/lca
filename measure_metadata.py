import h5py
from pathlib import Path
import sys

path_to_images = Path(r"Z:\20210930_dummy\hdf5")
path_to_features = Path(r"Z:\20210930_dummy\features")

# define the site this job should process
site = int(sys.argv[1])

# load hdf5 file of site
file = h5py.File(path_to_images.joinpath('site_%04d.hdf5' % site), "a")

# define settings

settings = {'nuclei': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'], 'assigned_object':'nuclei', 'aggregate':False},
            'cells': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'], 'assigned_object':'cells', 'aggregate':False},
            'nuclear_speckles': {'channels':['sdcGFP', 'sdcDAPIxmRFPm'], 'assigned_object':'cells', 'aggregate':False}}

for object_id, object in enumerate(settings.keys()):

    label_images = file['label_images/%s' % object][:]

    metadata = measure_metadata(label_images)

    # add a unique identifier for every object, currently allows for 10000 occurences per object per site
    metadata['unique_id'] = np.arange(0, len(metadata))
    metadata['unique_id'] = metadata['unique_id']+(site*len(settings.keys())*10000)+10000*object_id
    metadata = metadata.set_index('unique_id')

    # save feature values for this site
    metadata.to_csv(path_to_features.joinpath('site_%04d_%s_metadata.csv' % (site, object)))