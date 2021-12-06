#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:36:52 2019

@author: adrian tschan
"""

from distutils.core import setup

setup(name='lca',
      version='1.0',
      description='some modules for live cell analysis',
      author='Adrian Tschan',
      author_email='adrian.tschan@uzh.ch',
      packages=['lca', 'lca.nd', 'lca.ndt'],
	  package_dir={},
      install_requires=[
            'numpy',
            'pandas',
            'scipy>=1.4.1',
            'networkx',
            'scikit-image',
            'progress',
            'cellpose[all]',
            'cellpose-napari',
            'scikit-learn',
            'h5py',
            'imageio',
            'matplotlib',
            'progress',
            'scipy',
            'sklearn',
            'pyyaml',
            'napari-feature-visualization @ git+https://github.com/adrtsc/napari-feature-visualization',
            'napari-blob-detection @ git+https://github.com/adrtsc/napari-blob-detection',
            #'sslap @ git+https://github.com/OllieBoyne/sslap.git',
      ]
      )
