# live cell analysis (lca) python tool

This repository contains code for the analysis of 2DT and 3DT live imaging data. Typically, the processing of such data in this framework consists of the following steps:

<ol>
  <li> compression of the images into hdf5 format and illumination correction  </li>
  <li> segmentation of objects in the images </li>
  <li> measurement of features (e.g. morphology features or intensity features) based on the segmentations </li>
</ol>

## Installation

You will need to install everything on the cluster as well as locally (to do some parameter tuning).

<ol>
  <li> Create a virtual python environment </li>
  
    conda create -n lca python=3.9
  
  <li> Activate your virtual python environment </li>
  
    conda activate lca

  <li> Install lca from github </li>
  
    pip install git+https://github.com/adrtsc/lca.git

  <li> The tracking in this framework is formulated as a linear assignment problem. To solve this problem we use a very useful Cython implementation of the auction algorithm (https://github.com/OllieBoyne/sslap).
    
    pip install git+https://github.com/OllieBoyne/sslap.git
