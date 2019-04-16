# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# for OF
from PIL import Image
import time
import argparse
#import pyflow
from pyflow import pyflow
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
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'
# Some constant for the script
N = 1
DET = 'YOLO_IOU'
TASK = 'IOU'
WEEK = 'week5'
SEQ = 'S03'
CAM = 'c010'

DET_GAP = 5
PLOT_FLAG = False
SAVE_FLAG = True
OF_FLAG = False
REFINE = False
# OF
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

def histAnimation(det_file,save_in = None,cam = 'c010',track_id =None):
    """
    Input:
    movie_path
        If movie path is a movie - load and extract frame - # TODO:
        If movie path is a folder - load images
    det_file : xml,txt or pkl
        panda list with the following column names
            ['frame','track_id', 'xmax', 'xmin', 'ymax', 'ymin']
    save_in
        folder to save output frames / movie -# TODO:

    """

    # Get BBox detection from list
    df = ut.getBBox_from_gt(det_file)
    if track_id is None:
        track_id = pd.unique(df['track_id'])
    # Create folders if they don't exist
    if save_in is not None and not os.path.isdir(save_in):
        os.mkdir(save_in)



    df.sort_values(by=['frame'])


    # create trajectory of all track
    df_track = df.groupby('track_id')
    index = np.linspace(0,255,31)
    colors = plt.cm.hsv(index / float(max(index)))
    for id,tt in df_track:
        first_frame = True
        print('Track id {}'.format(id))
        if id not in track_id:

            continue


        print('frames:')
        for t in tt.index.tolist():
            # 1st frame -
            ts = tt.loc[t,'time_stamp']
            f = tt.loc[t,'frame']
            print(f)
            hist = tt.loc[t,'histogram']
            #print(hist)
            if first_frame:
                first_frame = False
                plt.ion()
                plt.show()
                fig = plt.figure()
                fig.suptitle('Camera {}, Track Id {}'.format(cam, id), fontsize = 16)
                ax = fig.add_subplot(111)

                ax.bar(range(len(hist)), hist, color = colors)
                #ax.bar(index, hist)
                ax.set_xlabel('Hue', fontsize=5)
                ax.set_ylabel('Probability', fontsize=5)
                #ax.set_xticklabels(range(len(hist)),index)
                ax.set_title('time {}'.format(ts))
            else:
                ax.clear()
                ax.bar(range(len(hist)), hist, color = colors)
                #ax.bar(index, hist)
                ax.set_xlabel('Hue', fontsize=5)
                ax.set_ylabel('Probability', fontsize=5)
                #ax.set_xticklabels(range(len(hist)),index)
                ax.set_title('time {}'.format(ts))

                if save_in is not None:

                    fig.savefig(os.path.join(save_in,'tk{}_f{}.png').format(id,f), dpi=fig.dpi)


    return

def main():
    """
    Add documentation.

    :return: Nothing
    """
    det_file = '../output/week5/HIST/det_yolo3S03c012_GT_histogram.pkl'
    cam = 'c012'
    output_path = os.path.join('../output/week5/HIST',cam)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    histAnimation(det_file,save_in = output_path,cam=cam,track_id =[243,246])

            # Read BBox from 1st Frame
            #Bbox_picked = ut.non_max_suppression(bboxes, overlap_thresh)

if __name__ == '__main__':
    main()
