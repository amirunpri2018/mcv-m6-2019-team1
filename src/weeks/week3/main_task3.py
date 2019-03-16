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
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../'
# Some constant for the script
N = 0.01
GT = 'no'
DIM = 3
EXP_NAME = '{}GT_N{}_DIM{}'.format(GT, N, DIM)
TASK = 'task3'
WEEK = 'week2'


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

    print(gt_file)

    # Get BBox detection from list
    df = ut.get_bboxes_from_MOTChallenge(gt_file)
    df.sort_values(by=['frame'])
    df.loc[:,'track_id'] = -1
    #df['track_id'] = -1
    #print(df)
    #print('========@@@@@@@@@@@@@@@@@@')
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
    df_track = pd.DataFrame(columns=headers)
    #df_track.assign(track_id=[])

    #Initialize Track ID - unique ascending numbers
    Track_id = 8


    for f, df_group in df_grouped:

        im_path = os.path.join(frames_dir,'frame_'+str(f).zfill(3)+'.jpg')

        # 1st frame -
        if frame_p ==0:

            frame_p = df_group['frame'].values[0]
            print('First detected object at frame {}'.format(frame_p))
                    # Assign new tracks
            for t in range(len(df_group)):
                df_group.at[t, 'track_id'] = Track_id
                Track_id+=1
            #df_group = df_group.assign(track_id=range(Track_id,Track_id+len(df_group)))
            df_p_group = pd.DataFrame(columns=headers)
            df_p_group = df_p_group.append(df_group, ignore_index=True)

            #df_p_group = df_group
            #df_p_group = df_p_group.assign(track_id=range(Track_id,Track_id+len(df_group)))
            Track_id +=len(df_group)
            # for t,df_object in enumerate(df_group):
            #     # Give Track_id to each object in the first frame
            #     df_object['track_id'] = Track_id
            #     Track_id +=1
            #     df_p_group = df_track.append(df_object, ignore_index=True)
            df_track = df_track.append(df_p_group, ignore_index=True)
            #print(df_p_group)
            #print('<<<<<<<<<<<<<<<<<')
                #df_gt_p = df_object[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()


            #plot 1st frame
            plt.ion()
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111)

            bbox =  bb.bbox_list_from_pandas(df_p_group)



            ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=df_p_group['track_id'].tolist(),ax=ax,title = str(f))
            #ut.plot_bboxes(ax ,img,bbox,l= labels, title=str(frame_p))

                    #df = df.append({'A': i}, ignore_index=True)
            continue




        frame_p = df_group['frame'].values[0]

        iou_mat = bb.bbox_lists_iou(df_p_group,df_group)
        pp = np.shape(iou_mat)
        if pp[0]>1 and pp[1]>1:
            print(iou_mat)
            print(matchlist)
            print(offset)

        matchlist = bb.match_iou(iou_mat,iou_th=0)

        # sort it according to the new frame
        offset = np.min(df_group.index.values.tolist() )

        for t in matchlist:

            df_group.at[offset+t[1], 'track_id'] = df_p_group.get_value(t[0],'track_id')


            # Setting the confidence as the iou score
            # TODO

            #if df_p_group.get_value(t[0],'frame') ==np.nan:
                #print(df_p_group)

        # Assign new tracks
        for t in df_group.index[df_group['track_id'] == -1].tolist():

            df_group.at[t, 'track_id'] = Track_id
            Track_id+=1

        df_track = df_track.append(df_group, ignore_index=True)

        df_p_group = pd.DataFrame(columns=headers)
        df_p_group = df_p_group.append(df_group, ignore_index=True)
        df_p_group = df_p_group.dropna()
        bbox =  bb.bbox_list_from_pandas(df_p_group)
        #print(df_p_group['track_id'].tolist())
        ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=df_p_group['track_id'].tolist(),ax=ax,title = str(f))
        #bbox_iou(bboxA, bboxB)


    # Read BBox from 1st Frame
    #Bbox_picked = ut.non_max_suppression(bboxes, overlap_thresh)

if __name__ == '__main__':
    main()
