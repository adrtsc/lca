import yaml

img_path = r'/data/active/atschan/20210930_dummy/MIP'
hdf5_path = r'/data/active/atschan/20210930_dummy/hdf5'
feature_path = r'/data/active/atschan/20210930_dummy/features'
illcorr_path = r'/data/active/atschan/illumination_correction/'
temp_seg_path = r'/data/active/atschan/20210930_dummy/temp_segmentation'
magnification = 40
file_extension = 'tif'
illumination_correction = False
output_filename = 'test'

settings = {'paths': {'img_path': img_path,
                      'hdf5_path': hdf5_path,
                      'feature_path': feature_path,
                      'illcorr_path': illcorr_path,
                      'temp_seg_path': temp_seg_path},
            'file_extension': file_extension,
            'magnification': magnification,
            'illumination_correction': illumination_correction
            'spacing': [0, 0, 1],
            'objects': {'nuclei': {'measure_morphology': True,
                                   'measure_intensity': ['sdcGFP', 'sdcDAPIxmRFPm'],
                                   'measure_blobs': {'channels': ['sdcGFP'],
                                                     'settings': {'min_sigma': 2,
                                                                  'max_sigma': 10,
                                                                  'num_sigma': 10,
                                                                  'threshold': 0.00015,
                                                                  'overlap': 0.5,
                                                                  'exclude_border': True}},
                                   'measure_tracks': {'max_distance': 100,
                                                      'time_window': 3,
                                                      'max_split_distance': 150,
                                                      'max_gap_closing_distance': 150,
                                   'assigned_objects': [],
                                   'segmentation_channel': 'sdcDAPIxmRFPm'},
                      'cells': {'measure_morphology': True,
                                'measure_intensity': ['sdcGFP', 'sdcDAPIxmRFPm'],
                                'measure_blobs': {'channels': ['sdcDAPIxmRFPm'],
                                                      'settings': {'min_sigma': 2,
                                                                   'max_sigma': 10,
                                                                   'num_sigma': 10,
                                                                   'threshold': 0.00015,
                                                                   'overlap': 0.5,
                                                                   'exclude_border': True}},
                                'measure_tracks': True,
                                'assigned_objects': ['nuclei', 'cytoplasm'],
                                'segmentation_channel': 'sdcDAPIxmRFPm'},
                      'cytoplasm': {'measure_morphology': True,
                                    'measure_intensity': ['sdcGFP', 'sdcDAPIxmRFPm'],
                                    'measure_blobs': {'channels': ['sdcDAPIxmRFPm'],
                                                      'settings': {'min_sigma': 2,
                                                                   'max_sigma': 10,
                                                                   'num_sigma': 10,
                                                                   'threshold': 0.00015,
                                                                   'overlap': 0.5,
                                                                   'exclude_border': True}},
                                    'measure_tracks': False,
                                    'assigned_objects': []}}},
            'channel_colors': {'sdcDAPIxmRFPm': 'blue',
                               'sdcGFP': 'green',
                               'sdcYFP': 'yellow',
                               'sdcRFP590': 'red',
                               'sdcCy5': 'magenta'}}


with open(f'scripts/settings/{output_filename}.yml', 'w') as outfile:
    yaml.dump(settings, outfile, default_flow_style=False)
