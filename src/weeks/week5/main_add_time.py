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
VID_FLAG = False
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

def main():
    """
    Add documentation.

    :return: Nothing
    """
    SEQ_LIST = os.listdir(os.path.join(ROOT_DIR,'train'))
    for SEQ in SEQ_LIST:
        seq_dir = os.path.join(ROOT_DIR,'train',SEQ)
        CAMER_LIST = os.listdir(seq_dir)

        for CAM in CAMER_LIST:
            gt_dir = os.path.join(seq_dir,CAM)


            fname = os.path.join(results_dir,"gt.pkl")
            if not os.path.isfile(fname):
                continue

            time_offset, fps = ut.obtain_timeoff_fps(ROOT_DIR,SEQ, CAM)
            print('Seq :{}, Camera;{} has on offset of {} sec, and {} fps'.format(SEQ,CAM,time_offset,fps))

            df = ut.getBBox_from_gt(fname)

            df.sort_values(by=['frame'])


        # New columns for tracking

        # Motion

            df.loc[:,'Mx'] = -1.0
            df.loc[:,'My'] = -1.0
            df.loc[:,'area'] = -1.0
            df.loc[:,'ratio'] = -1.0
            df.loc[:,'time_stamp'] = 0.0
            # Group bbox by frame
            df_grouped = df.groupby('frame')

            vals = list()

            # Get first bbox

            frame_p = 0
            df_gt_p = []

            # iterate over each group
            #df_track = pd.DataFrame({'frame':[]:'ymin':[], 'xmin':[], 'ymax':[], 'xmax':[]})
            headers = list(df.head(0))
            print(headers )
            print('---------')
            df_track = pd.DataFrame(columns=headers)

            for f, df_group in df_grouped:
                df_group = df_group.reset_index(drop=True)
                if f%50==0:
                    print(f)


                    offset = np.min(df_group.index.values.tolist() )

                    df_group.loc[[offset+t[1]],'time_stamp'] = ut.timestamp_calc(f, time_offset, fps)
                        #print(df_group)


                        # Setting the confidence as the iou score
                        # TODO
                else:
                    print('All Tracks were initialized becaue there was no detection fot {} frames'.format(DET_GAP))

                    #print(df_group)
                # Assign new tracks
                for t in df_group.index[df_group['track_id'] == -1].tolist():
                    bAc,bAa, bAr = bb.getBboxDescriptor(df_group.ix[[t]])
                    df_group.at[t, 'track_id'] = Track_id
                    df_group.at[t,'Mx'] = bAc[0]
                    df_group.at[t, 'My'] = bAc[1]
                    df_group.at[t, 'area'] = bAa
                    df_group.at[t, 'ratio'] = bAr
                    df_group.at[t,'time_stamp'] = ut.timestamp_calc(f, time_offset, fps)
                    Track_id+=1


                #print(df_group)
                df_p_group = pd.DataFrame(columns=headers)
                df_p_group = df_p_group.append(df_group, ignore_index=True)
                df_p_group = df_p_group.dropna()
                #print(df_p_group)
                df_track = df_track.append(df_p_group, ignore_index=True)


            if SAVE_FLAG:
                save_in = os.path.join(results_dir,"pred_tracks.pkl")
                df_track.to_pickle(save_in)
                csv_file = os.path.splitext(save_in)[0] + "1.csv"
                export_csv = df_track.to_csv(csv_file, sep='\t', encoding='utf-8')

            # Read BBox from 1st Frame
            #Bbox_picked = ut.non_max_suppression(bboxes, overlap_thresh)

if __name__ == '__main__':
    main()
