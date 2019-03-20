# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# For Metric
#from sklearn.metrics import average_precision_score
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# average_precision_score(y_true, y_scores)
# For visulization

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# import List libariry
import pandas as pd
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
import metrics.video as m
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../'
# Some constant for the script
N = 1
DET = 'GT'
EXP_NAME = '{}_N{}'.format(DET, N)
TASK = 'task2'
WEEK = 'week3'
DET_GAP = 5
PLOT_FLAG = True
VID_FLAG = True
SAVE_FLAG = True


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
    gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation.pkl')

    # detection file can be txt/xml/pkl
    det_file = os.path.join(results_dir,'kalman_out.pkl')

    # if the result matrix - which assigned each track and bbox to TP,FP,FN
    # no need to load all the data
    res_file = os.path.join( results_dir,'result_mat.pkl')
    if os.path.isfile(res_file):
        result_list = pd.read_pickle(res_file)
    else:
        # Get BBox detection from list
        df_gt = ut.getBBox_from_gt(gt_file)
        df_det = ut.getBBox_from_gt(det_file)

        result_list = m.PandaTpFp(df_det,df_gt ,iou_thresh = 0.5,save_in =res_file)


    # Computing mAP for detections
    mAP=m.PandaTo_PR(result_list)
    print('mAP detection :{}'.format(mAP))
    mAP_track,Ap_perTrack=m.PandaTrack_PR(result_list)
    print('mAP tracking :{}'.format(mAP_track))
    Ap_perTrack = np.asarray(Ap_perTrack)
    n_bins = 10
    bins_h = np.linspace(np.min(Ap_perTrack), np.max(Ap_perTrack), num=n_bins)
    #val_hist, bin_centers = ut.histogram(np.asarray(Ap_perTrack), bins=len(Ap_perTrack)/10)
    fig = plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.hist(Ap_perTrack.ravel(), bins=bins_h,alpha=0.65) #,width=0.8
    ax1.set(xticks=bins_h)
    ax1.locator_params(axis='x', nbins=5)

    ax1.axvline(mAP_track, color='r', linestyle='dashed', linewidth=2)
    _, max_ = plt.ylim()
    ax1.text(mAP_track + mAP_track/20, max_ - max_/20,'Mean: {:.2f}'.format(mAP_track))

    ax1.set_title('mAP over tracking')
    plt.show()
    plt.savefig(os.path.join(results_dir,'mAP_track.png'))
