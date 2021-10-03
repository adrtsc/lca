#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:36:52 2019

@author: team1
"""

from distutils.core import setup

setup(name='lca',
      version='1.0',
      description='Some submodules for live cell analysis',
      author='Adrian Tschan',
      author_email='adrian.tschan@uzh.ch',
      packages=['lca'],
	  package_dir={'': 'src'},
     )