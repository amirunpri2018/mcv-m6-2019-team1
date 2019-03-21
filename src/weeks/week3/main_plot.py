# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# For visulization

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# import List libariry
import pandas as pd
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut

# Visualization
#import visualization.utils as ut_vis

import evaluation.bbox_iou as bb
import metrics.video as m
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../'
# Some constant for the script
N = 1
DET = 'YOLO_refine'
EXP_NAME = '{}_N{}'.format(DET, N)
TASK = 'task3'
WEEK = 'week3'
DET_GAP = 5
PLOT_FLAG = True
VID_FLAG = False
SAVE_FLAG = True
REFINE = False

def main():
    """
    Add documentation.

    :return: Nothing
    """

    # Set useful directories
    frames_dir = os.path.join(
        ROOT_DIR,
        'data',
        'm6_week1_frames',
        'frames')
    results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)

    # Create folders if they don't exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    # Ground truth file path
    gt_file = os.path.join(ROOT_DIR,
                           'data', 'AICity_data', 'train', 'S03',
                           'c010', 'gt', 'gt.txt')

    #gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation.xml')
    gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation2.pkl')
    gt = pd.read_pickle(gt_file)
    print(len(gt))
    # remove bicycles
    gt = gt[gt.label =='car']
    #mask = gt.groupby('label').label.transform('values') == 'car'

    #gt =gt[mask]
    #gt.to_pickle(os.path.join(ROOT_DIR,'data', 'm6-full_annotation3.pkl'))
    #print(len(gt))
    #headers = list(gt.head(0))
    #print(headers )
    #print('---------')
    # detection file can be txt/xml/pkl
    #det_file = os.path.join(results_dir,'kalman_out.pkl')
    det_file = os.path.join(results_dir,'pred_tracks.pkl')
    # if the result matrix - which assigned each track and bbox to TP,FP,FN
    # no need to load all the data
    res_file = os.path.join( results_dir,'result_mat.pkl')

    if SAVE_FLAG:
        ut.bboxAnimation(frames_dir,det_file,save_in = results_dir)
    else:
        ut.bboxAnimation(frames_dir,det_file)
