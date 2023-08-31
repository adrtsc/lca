# live cell analysis (lca) python tool

<img src="https://github.com/adrtsc/lca/blob/3D/logo/lca_logo.png?raw=true" width="250" title="lca" alt="lca" align="right" vspace = "100">

This repository contains code for the analysis of 2DT and 3DT live imaging data. Typically, the processing of such data in this framework consists of the following steps:

<ol>
  <li> compression of the images into hdf5 format and illumination correction  </li>
  <li> segmentation of objects in the images </li>
  <li> measurement of features (e.g. morphology features or intensity features) based on the segmentations </li>
</ol>

For each of these steps, the user will have to define suitable parameters. These parameters are then saved in a settings file that is specific for each experiment and which will be used in the final image processing.

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
    
## Typical Workflow

### Generation of a settings file

A settings yaml file is neccessary that stores the information about the experiment. This includes where the raw images are saved, where the output should be generated, what kind of measurements should be made and other things. It's easiest to use one of the example settings file located in the /scripts/settings folder and adapt it to your needs.

### Initialize zarr arrays

The first step of the workflow is to convert the image data into zarr arrays. To intialize the zarr arrays you need to have generated the settings file and then run the following:

    python initialize_zarr.py path/to/settings_file.yaml
    
This will generate empty zarr arrays in the output directory. The zarr files will be organized like this:
 
     zarr
        -intensity_images
            -channel_00
            -channel_01
            -channel_03
                -level_0
                -level_1
                -level_2
                -level_3
          
Where the "intensity_images" group contains another group for every channel that was acquired. Inside the channel group the 4D datasets are contained (one for every level of a multiscale pyramid).
 
### Compress image files
 
After initializing the zarr arrays, the image data can be compressed and saved into them. To do so, run the following line on a cluster:
 
    python 00_run_zarr_compression.py /path/to/settings_file.yaml

This will start a job on the slurm cluster for every timepoint/site (by default timepoint) for the image data. The image data will be compressed into the zarr format and saved into the zarr files that were initialized previously.

### Object segmentation

Currently, only the cellpose model for nuclei and cell segmentation is supported. To run the segmentation (based on intensity images define in the settings file) run the following code:

    python 01_run_cellpose_zarr.py /path/to/settings_file.yaml
    
This will generate a new group on the same level as "intensity_images" called "label_images". It is organized the same way as the intensity images, only that each object that is segmented is represented by another group (instead of each channel in the "intensity_images" group.

### Feature measurements

After generating the segmentations, the features can be measured. To do this run the following line:

    python 03_run_measure_features.py /path/to/settings_file.yaml
