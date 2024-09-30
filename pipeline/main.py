# This module serves as a pipeline that takes
# input selections and output a 3d reconstruction
# The module is supposed to load the algorithms 
# chosen in the CONFIG.yaml to accomplish the functionality
# The code is organized by 'region' and 'endregion' 
# Form should be optimized for VSCode IDE
'''
Author: Yicheng Zhu
RoCAL
Instructor: Dr. Yangming Lee
'''
# region import
import os
import shutil
from dataloader import yaml_handler, frame_handler, temp_folder
from depthEstimate import estimate as depth
from visualize import outputvisual as op
# endregion

# region Configure Reading
'''
This part of the code is for reading the configuration to the program.

Configuration is written in the form of a yaml file, and the dictionary-like
structure should be strictly followed to maintain the readability of the yaml.

TODO: A setup.py should be written to handle the editing of the CONFIG.yaml

The CONFIG.yaml is structured as 
Data # Everything related to the data used for this pipeline.
    -DATA_PATH # Can be a video, can be a folder
    -DATA_TYPE # Well, either 'video' or 'folder' #TODO: This doesn't seem necessary, can be removed in the future
------------------Scheme chosen decide what kind of operation will be done-----
Scheme
    -SELECT_SCHEME # This can be a list of methods, executed in order, to select frames from the source.
------------------The parameters used by the selected Schemes------------------
Color
    -R_THRESHOLD # If Color-r is chosen, threshold for judging if a frame is Red enough
Hyper-IQA
    -IQA_THRESHOLD # If Hyper IQA is chosen, threshold for judgin if frame is low quality
------------------Irrelevant to the operation, just options for visualiation---
Visual
    -IF_MUTE # Default to be True, can be set to False to help keep track of
    intermediate results.
'''
# region CODE
# Loading all Configuration from the yaml file
file_path = 'pipeline/CONFIG.yaml'
config = yaml_handler.load_yaml(file_path)
if config:

    op.block_line_output('Configuration Successfully Loaded')

# If Muting
mute = config['IF_MUTE']
# endregion
# endregion

# region Data Preparation
'''
This part of the code handles the data preparation 
Will be skipped if the data doees not need to be processed, of course.
The functionalities include getting a frame list which contains all 
absolute paths to the frames we would like to use.
'''
# region CODE
datatype = config['DATA_TYPE']
if datatype == 'video':
    #do something to prepare the video to folder
    pass
datapath = config['DATA_PATH']
frameList = frame_handler.getFrameList(datapath, mute)

op.block_line_output('Data Preparation Done.')
# endregion
# endregion

# region Frame Selection
'''
This part of code handles the First step of operation
The first actual step of the pipeline is to pick out valid frames from a big 
bunch of frames. Some of them are usable and some are not. 
However, how shall we pick the frames is a big question mark.
Some frames clearly should be picked out:
-- The frames that are not on-site. 
    --I believe that they can be picked out by thresholding the red channel,
    since the hue of human tissue and the intensity of inside the patient and 
    outside should be different
-- The frames that are blurry/low-quality.
    --Image quality assessment can be used to pick them out. 
    --TODO: HyperIQA with the pretrained 'koniq_pretrained' model
    is probably not the one we should use. Seems like it 
    scores the images from Hamlyn dataset to be around 40~/100. 
-- The frames that are TODO: list out more kinds of frames that we can pick 
    out using other frame-selection methods. 
'''
# region CODE
# Load method selection
selection_method = config['SELECT_SCHEME']
method = None

for method_i in range(len(selection_method)):
    current_method = selection_method[method_i]
    if current_method == 'quality':
        q_method = config['Q_METHOD']
        if q_method == 'hyperIQA':
            from frameSelect.hyperIQA import hyperIQA
            method = hyperIQA.hyperIQA()        
            # Use the method and get a list of scores
            selection_score = method.predict_list(frameList, mute = mute)
            IQA_threshold = config['IQA_THRESHOLD']
            
            # Once we have the selection score, apply thresholding to choose the 
            # qualified frames from the framelist
            frameList = frame_handler.pickFrameList(frameList, selection_score,IQA_threshold)
            op.block_line_output('Hyper IQA applied and frameList have been updated.')
            method = None
    elif current_method == 'color-r':
        op.block_line_output('Color-Red is chosen but not implemented yet.')
    elif current_method == 'feature':
        op.block_line_output('Feature-based selection is chosen but not implemented yet.')
    else:
        op.block_line_output('No valid frame selection method is chosen.')

# endregion
# endregion
op.block_line_output('---END OF FRAME SELECTION---')
# region Temp Folder Creation
'''
A temporary folder is created to be the folder of chosen frames.
This serves several purposes: 
-keep track of which frames are chosen per iteration.
-some algorithms are more easily used when passing a folder. 
'''
# region CODE
temp_path = temp_folder.create(frameList, mute)
# endregion
# endregion
# region Depth Estimation
'''
Once we have the frames picked out, we are going foward and 
make depth estimation for each of them. The depth map is essential 
for the 3D reconstruction.
If depth estimation is not needed, a depth map path will be read and 
the valid depth map will be pulled from that folder depending on the 
FrameList.
'''
# Depth Estimation

depth_method = config['DEPTH_SCHEME']


# After picking the depth generating method, the framelist will be 
# taken and it shall be sent as input for the depth estimation
if depth_method == 'depth-anything':
    op.block_line_output('Depth-anything is the chosen depth prediction method.')
    depth_anything_config = [config['DA_ENCODER'],config['PRED_ONLY'],config['GRAY_SCALE']]
    depth.estimate(depth_method,temp_path,depth_anything_config, mute)


# 3D reconstruction

# Release Temp folder
temp_folder.clear()

op.end()