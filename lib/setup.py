#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:36:52 2019

@author: team1
"""

from distutils.core import setup

setup(name='lca',
      version='1.0',
      description='some modules for live cell analysis',
      author='Adrian Tschan',
      author_email='adrian.tschan@uzh.ch',
      packages=['lca'],
	  package_dir={'': 'src'},
      install_requires=[
      'numpy',
      'pandas',
      'scipy>=1.4.1',
      'networkx',
      'scikit-image',
      'progress',
      'cellpose',
      'sslap',
      'scikit-learn',
      'h5py',
      'imageio',
      'matplotlib',
      'opencv',
      'progress',
      'scipy',
      'sklearn',
      ]
     )