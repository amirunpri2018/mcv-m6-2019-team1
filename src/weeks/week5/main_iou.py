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
TASK = 'IOU_S03'
WEEK = 'week5'
SEQ = 'S04'
CAM = 'c010'

DET_GAP = 5
PLOT_FLAG = False
VID_FLAG = False
SAVE_FLAG = True
OF_FLAG = False
REFINE = True
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
    DATA_ROOT = ROOT_DIR#os.path.join(ROOT_DIR,'data', 'AICity_data')
    # Set useful directories
    #CAMS = ['c016','c017','c018','c019','c020']
    CAMS = os.listdir(os.path.join(DATA_ROOT, 'train', SEQ))
    for CAM in CAMS:
        EXP_NAME = '{}_{}_{}_N{}'.format(SEQ,CAM,DET, N)
        frames_dir = os.path.join(DATA_ROOT, 'train', SEQ,
                               CAM,'frames')
        results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)
        # Create folders if they don't exist
        if not os.path.isdir(os.path.join(OUTPUT_DIR, WEEK, TASK)):
            os.mkdir(os.path.join(OUTPUT_DIR, WEEK, TASK))
        # Create folders if they don't exist
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        # Ground truth file path
        gt_file = os.path.join(DATA_ROOT, 'train', SEQ,
                               CAM, 'gt', 'gt.txt')


        #gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation.xml')
        #gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation2.pkl')
        # get camera offsets
        time_offset, fps = ut.obtain_timeoff_fps(DATA_ROOT,SEQ, CAM)
        print('Seq :{}, Camera;{} has on offset of {} sec, and {} fps'.format(SEQ,CAM,time_offset,fps))
        #df = ut.getBBox_from_gt(gt_file,save_in = out_file)
        #df = ut.getBBox_from_gt(gt_file)
        #print(gt_file)

        det_file = os.path.join(ROOT_DIR, 'train',SEQ,
                               CAM, 'det', 'det_yolo3.txt')
        # Get BBox detection from list
        #det_file = gt_file
        df = ut.getBBox_from_gt(det_file)
        #print(df.dtypes)
        #df = ut.get_bboxes_from_MOTChallenge(gt_file)
        # Get BBox from xml gt_file
        #df = get_bboxes_from_aicity_file(fname, save_in=None)
        df.sort_values(by=['frame'])

        df.loc[:,'track_id'] = -1
        # New columns for tracking

        df.loc[:,'track_iou'] = -1.0
        # Motion
        df.loc[:,'Dx'] = -300.0
        df.loc[:,'Dy'] = -300.0
        df.loc[:,'rot'] = -1.0
        df.loc[:,'zoom'] = -1.0
        df.loc[:,'Mx'] = -1.0
        df.loc[:,'My'] = -1.0
        df.loc[:,'area'] = -1.0
        df.loc[:,'ratio'] = -1.0
        df.loc[:,'ofDx'] = 0.0
        df.loc[:,'ofDy'] = 0.0
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


        #Initialize Track ID - unique ascending numbers
        Track_id = 8


        for f, df_group in df_grouped:
            df_group = df_group.reset_index(drop=True)
            if f%50==0:
                print(f)
            if f>3000:
                break

            im_path = os.path.join(frames_dir,'frame_'+str(int(f)).zfill(3)+'.jpg')

            # 1st frame -
            if frame_p ==0:

                frame_p = df_group['frame'].values[0]
                print('First detected object at frame {}'.format(frame_p))


                        # Assign new tracks
                for t in range(len(df_group)):

                    #print(df_group.loc['track_id'])
                    df_group.at[t, 'track_id'] = Track_id
                    bAc,bAa, bAr = bb.getBboxDescriptor(df_group.ix[[t]])
                    df_group.at[t,'Mx'] = bAc[0]
                    df_group.at[t, 'My'] = bAc[1]
                    df_group.at[t, 'area'] = bAa
                    df_group.at[t, 'ratio'] = bAr
                    df_group.at[t, 'time_stamp'] = ut.timestamp_calc(f, time_offset, fps)
                    Track_id+=1

                df_p_group = pd.DataFrame(columns=headers)
                df_p_group = df_p_group.append(df_group, ignore_index=True)
                df_p_group = df_p_group.dropna()
                Track_id +=len(df_group)

                df_track = df_track.append(df_p_group, ignore_index=True)
                continue




            frame_p = df_group['frame'].values[0]

            # if there is more than N frames between detection - it is a new track - even if it in the bbox overlaps

            if df_p_group['frame'].values[0]+DET_GAP >df_group['frame'].values[0]:

                iou_mat = bb.bbox_lists_iou(df_p_group,df_group)
                #print('first',iou_mat)
                matchlist,iou_score = bb.match_iou(iou_mat,iou_th=0)
                #print('second',iou_mat)
                #print(iou_score)
                # sort it according to the new frame

                offset = np.min(df_group.index.values.tolist() )
                #print(offset)
                if OF_FLAG:
                    bbox =  bb.bbox_list_from_pandas(df_group)
                    im_path_curr = os.path.join(frames_dir,'frame_'+str(int(f)).zfill(3)+'.jpg')
                    im_path_prev = os.path.join(frames_dir,'frame_'+str(int(f)-1).zfill(3)+'.jpg')
                    print(im_path_prev)
                    print(im_path_curr)
                    im1 = np.array(Image.open(im_path_curr))
                    im2 = np.array(Image.open(im_path_prev))
                    im1 = im1.astype(float) / 255.
                    im2 = im2.astype(float) / 255.
                    u, v, im2W = pyflow.coarse2fine_flow(
                            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                            nSORIterations, colType)

                for t,iou_s in zip(matchlist,iou_score):
                    df_group.at[offset+t[1], 'track_id'] = df_p_group.get_value(t[0],'track_id')
                    df_group.at[offset+t[1], 'track_iou'] = iou_s
                    # Motion parameters
                    #------------------
                    # Dx ,Dv - of bbox center
                    # Zoom - Ratio of areas
                    # Rot - Ratio of Ratio(h/w)'' -describes rotation
                    if OF_FLAG:
                        c_bbox = bbox[t[1]]
                        xv, yv = np.meshgrid(range(int(c_bbox[1]),int(c_bbox[3])),range(int(c_bbox[0]),int(c_bbox[2])))

                        c_u = u[ yv,xv]
                        c_v = v[yv,xv]
                    else:
                        c_u = 0.0
                        c_v = 0.0



                    box_motion = bb.getMotionBbox(df_p_group.ix[[t[0]]],df_group.ix[[offset+t[1]]])

                    df_group.loc[[offset+t[1]],'rot'] = box_motion[3]#.columns = ['Dy', 'Dx','zoom','rot','Mx','My']
                    df_group.loc[[offset+t[1]],'zoom'] = box_motion[2]
                    df_group.loc[[offset+t[1]],'Dx'] = box_motion[0]
                    df_group.loc[[offset+t[1]],'Dy'] = box_motion[1]
                    df_group.loc[[offset+t[1]],'Mx'] = box_motion[4]
                    df_group.loc[[offset+t[1]],'My'] = box_motion[5]
                    df_group.loc[[offset+t[1]],'area'] = box_motion[6]
                    df_group.loc[[offset+t[1]],'ratio'] = box_motion[7]
                    df_group.loc[[offset+t[1]],'ofDx'] = np.mean(c_u)
                    df_group.loc[[offset+t[1]],'ofDy'] = np.mean(c_v)
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


        #    END OF PROCESS

        if REFINE:
            #df_det = ut.track_cleanup(df_det,MIN_TRACK_LENGTH=10)
            #df_track = ut.track_cleanup(df_track,MIN_TRACK_LENGTH=10,MOTION_MERGE=5)
            save_in = os.path.join(results_dir,"pred_tracks0.pkl")
            df_track.to_pickle(save_in)
            df_track = ut.track_cleanup(df_track,MIN_TRACK_LENGTH=10,STATIC_OBJECT = 15) # object moved less than 15 pix


        print('Number of Detections:')
        print(np.shape(df_track))

        if SAVE_FLAG:
            save_in = os.path.join(results_dir,"pred_tracks.pkl")
            df_track.to_pickle(save_in)
            csv_file = os.path.splitext(save_in)[0] + "1.csv"
            export_csv = df_track.to_csv(csv_file, sep='\t', encoding='utf-8')

        # Read BBox from 1st Frame
        #Bbox_picked = ut.non_max_suppression(bboxes, overlap_thresh)

if __name__ == '__main__':
    main()
