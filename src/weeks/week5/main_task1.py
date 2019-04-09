# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# import List libariry
import pandas as pd
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut
#import organize_code.code.utils_opticalFlow as of
import organize_code.utilsBG as bg
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
# import pyopt
# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
# from __future__ import unicode_literals

from PIL import Image
import time
import argparse
#import pyflow
from pyflow import pyflow
# for Learning
from sklearn import linear_model
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'
# Some constant for the script
N = 1
DET = 'YOLO'

TASK = 'task1'
WEEK = 'week5'
SEQ = 'S03'
CAM = 'c010'
EXP_NAME = '{}_{}_{}_N{}'.format(SEQ,CAM,DET, N)

PLOT_FLAG = False
VID_FLAG = False
SAVE_FLAG = True
LEARN_FLAG = False
CLEAN_FLAG = False
TRACK_FLAG = True

def main():
    """
    Add documentation.

    :return: Nothing
    """


    # ONLY THE FALSE ALARMS and the TRACKING can be improved - but the missdetecion are lost - so the highest preciosion we can get has a ceiling
    #Initialize Track ID - unique ascending numbers

    # Set useful directories
    CAM_PATH = os.path.join(ROOT_DIR, 'train', SEQ, CAM)
    frames_dir = os.path.join(CAM_PATH,'frames')
    results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)

    gt_file = os.path.join(CAM_PATH, 'gt', 'gt.pkl')
    # Create folders if they don't exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    # detection file can be txt/xml/pkl
    det_file = os.path.join(ROOT_DIR,'train', SEQ,CAM, 'det', 'det_yolo3.txt')

    # Get BBox detection from list
    #df = ut.getBBox_from_gt(det_file)

    #df.sort_values(by=['frame'])

    frame_paths = ut.get_files_from_dir2(frames_dir, ext='.jpg')
    frame_paths.sort(key=ut.natural_keys)
    Nt = 300
    train_frames = frame_paths[:Nt]
    # Creating BG image from a mean of all seq
    mu_bg, std_bg = bg.getGauss_bg(train_frames, D=1)
    cv.imwrite(os.path.join(CAM_PATH,"BG.png"), mu_bg);


if __name__ == '__main__':
    main()
