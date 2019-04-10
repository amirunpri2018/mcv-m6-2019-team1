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
#import utils.image as ut_img
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
TASK = 'task_homography'
WEEK = 'week5'

SEQ = 'S01'
CAM1 = 'c002'
CAM2 = 'c004'
INPUT = 'det_mask_rcnn'
DET = 'det_mask_rcnn'
EXP_NAME = '{}_{}_{}_{}'.format(SEQ, CAM1, CAM2, DET)
GPS = [42.525678, -90.723601, 1]

# Flags:
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
    CAM1_PATH = os.path.join(ROOT_DIR, 'train', SEQ, CAM1)
    frames1_dir = os.path.join(CAM1_PATH,'frames')
    cam1_cal_file = os.path.join(CAM1_PATH,'calibration.txt')

    CAM2_PATH = os.path.join(ROOT_DIR, 'train', SEQ, CAM2)
    frames2_dir = os.path.join(CAM2_PATH, 'frames')
    cam2_cal_file = os.path.join(CAM2_PATH, 'calibration.txt')

    results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)

    #if not os.path.isdir(results_dir):
    #    os.mkdir(results_dir)

    #frame1_paths = ut.get_files_from_dir2(frames1_dir, ext='.jpg')
    #frame2_paths = ut.get_files_from_dir2(frames2_dir, ext='.jpg')
    #frame_paths.sort(key=ut.natural_keys)
    #frame1_paths.sort
    #frame2_paths.sort

    cal1_matrix = ut.get_cal_matrix(cam1_cal_file)
    cal2_matrix = ut.get_cal_matrix(cam2_cal_file)
    cal2_matrix_inv = np.linalg.inv(cal2_matrix)
    H_21 = np.dot(cal1_matrix, cal2_matrix_inv)

    gt1_file = os.path.join(CAM1_PATH, 'gt', 'gt.pkl')
    gt2_file = os.path.join(CAM2_PATH, 'gt', 'gt.pkl')

    gt1 = pd.read_pickle(gt1_file)
    gt2 = pd.read_pickle(gt2_file)

    cam1_off = 1.64
    cam2_off = 2.177  #Frme1 in cam2 will be frame 6 / 7 in cam 1

    px1 = (658.0, 672.0, 1) # (xmax, ymax, 1)
    #px1 = (725.0, 481.0, 1) # (xmax, ymax, 1)
    px2 = (1918.0, 471.0, 1) # (xmax, ymax, 1)

    px1_homog = np.dot(H_21, px2)
    px1_result = px1_homog / px1_homog[-1] # [ 88.93484793 270.61688663   1.        ]

    print(px1_result)

    #cv.imwrite(os.path.join(CAM_PATH,"BG.png"), mu_bg);


if __name__ == '__main__':
    main()
