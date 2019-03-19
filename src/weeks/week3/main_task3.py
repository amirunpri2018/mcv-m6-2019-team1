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
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../'
# Some constant for the script
N = 0.01
GT = 'no'
DIM = 3
EXP_NAME = '{}GT_N{}_DIM{}'.format(GT, N, DIM)
TASK = 'task3'
WEEK = 'week3'
DET_GAP = 5
PLOT_FLAG = True
VID_FLAG = False
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

    print(gt_file)

    # Get BBox detection from list
    df = ut.get_bboxes_from_MOTChallenge(gt_file)
    df.sort_values(by=['frame'])
    df.loc[:,'track_id'] = -1
    # New columns for tracking

    df.loc[:,'track_iou'] = -1.0
    # Motion
    df.loc[:,'Dx'] = -300.0
    df.loc[:,'Dy'] = -300.0
    df.loc[:,'rot'] = -1.0
    df.loc[:,'zoom'] = -1.0

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


    #Initialize Track ID - unique ascending numbers
    Track_id = 8


    for f, df_group in df_grouped:
        if f>3000:
            break

        im_path = os.path.join(frames_dir,'frame_'+str(f).zfill(3)+'.jpg')

        # 1st frame -
        if frame_p ==0:

            frame_p = df_group['frame'].values[0]
            print('First detected object at frame {}'.format(frame_p))
                    # Assign new tracks
            for t in range(len(df_group)):
                df_group.at[t, 'track_id'] = Track_id
                Track_id+=1

            df_p_group = pd.DataFrame(columns=headers)
            df_p_group = df_p_group.append(df_group, ignore_index=True)

            Track_id +=len(df_group)

            df_track = df_track.append(df_p_group, ignore_index=True)



            #plot 1st frame
            if PLOT_FLAG:

                plt.ion()
                plt.show()
                fig = plt.figure()
                ax = fig.add_subplot(111)

                bbox =  bb.bbox_list_from_pandas(df_p_group)

                ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=df_p_group['track_id'].tolist(),ax=ax,title = str(f))

                if SAVE_FLAG:
                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

                #cv.imshow(img)
                    if VID_FLAG:
                        fourcc = cv.cv.CV_FOURCC('M','J','P','G')

                        s = np.shape(img)
                        out = cv.VideoWriter(os.path.join(results_dir,"IOU.avi"),fourcc, 30.0, (s[0],s[1]))
                        out.write(img)
            continue




        frame_p = df_group['frame'].values[0]

        # if there is more than N frames between detection - it is a new track - even if it in the bbox overlaps

        if df_p_group['frame'].values[0]+DET_GAP >df_group['frame'].values[0]:

            iou_mat = bb.bbox_lists_iou(df_p_group,df_group)
            matchlist,iou_score = bb.match_iou(iou_mat,iou_th=0)

            # sort it according to the new frame
            offset = np.min(df_group.index.values.tolist() )

            for t,iou_s in zip(matchlist,iou_score):
                df_group.at[offset+t[1], 'track_id'] = df_p_group.get_value(t[0],'track_id')
                df_group.at[offset+t[1], 'track_iou'] = iou_s
                # Motion parameters
                #------------------
                # Dx ,Dv - of bbox center
                # Zoom - Ratio of areas
                # Rot - Ratio of Ratio(h/w)'' -describes rotation

                box_motion = bb.getMotionBbox(df_p_group.ix[[t[0]]],df_group.ix[[offset+t[1]]])

                df_group.loc[[offset+t[1]],'rot'] = box_motion[3]#.columns = ['Dy', 'Dx','zoom','rot']
                df_group.loc[[offset+t[1]],'zoom'] = box_motion[2]
                df_group.loc[[offset+t[1]],'Dx'] = box_motion[1]
                df_group.loc[[offset+t[1]],'Dy'] = box_motion[0]
                #print(df_group)


                # Setting the confidence as the iou score
                # TODO
        else:
            print('All Tracks were initialized becaue there was no detection fot {} frames'.format(DET_GAP))

            #print(df_group)
        # Assign new tracks
        for t in df_group.index[df_group['track_id'] == -1].tolist():

            df_group.at[t, 'track_id'] = Track_id
            Track_id+=1


        #print(df_group)
        df_p_group = pd.DataFrame(columns=headers)
        df_p_group = df_p_group.append(df_group, ignore_index=True)
        df_p_group = df_p_group.dropna()
        #print(df_p_group)
        df_track = df_track.append(df_p_group, ignore_index=True)
        #print(df_track)
        if PLOT_FLAG:
            bbox =  bb.bbox_list_from_pandas(df_p_group)
            #print(df_p_group['track_id'].tolist())
            ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=[df_p_group['track_id'].tolist(),df_p_group['track_iou'].tolist()],ax=ax,title = "frame: "+str(f))

            if SAVE_FLAG:
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
                #print(np.shape(img))
                cv.imwrite(os.path.join(results_dir,"{}.png".format(f)), img);
                #cv.imshow('test',img)
                if VID_FLAG:

                    out.write(img)
        #bbox_iou(bboxA, bboxB)

    #    END OF PROCESS
    if VID_FLAG:
        out.release()

    print(df_track)
    print(np.shape(df_track))

    if SAVE_FLAG:
        df_track.to_pickle(os.path.join(results_dir,"iou_tracks.pkl"))

    # Read BBox from 1st Frame
    #Bbox_picked = ut.non_max_suppression(bboxes, overlap_thresh)

if __name__ == '__main__':
    main()
