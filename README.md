# live cell analysis (lca) python tool

This repository contains code for the analysis of 2DT and 3DT live imaging data. Typically, the processing of such data in this framework consists of the following steps:

<ol>
  <li> compression of the images into hdf5 format and illumination correction  </li>
  <li> segmentation of objects in the images </li>
  <li> measurement of features (e.g. morphology features or intensity features) based on the segmentations </li>
</ol>

## Installation

<ol>
  <li> create a virtual python environment </li>
  
    conda create -n lca python=3.9

  <li> install lca from github </li>
  
    pip install git+https://github.com/adrtsc/lca.git
