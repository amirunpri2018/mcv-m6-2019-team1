# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local libraries
import organize_code.code.utils as ut
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
    # Group bbox by frame
    df_grouped = df.groupby('frame')

    vals = list()

    # Get first bbox
    # df_group=df_grouped[0]
    # frame_p = df_group['frame'].values[0]
    # df_gt_p = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
    # print('First detected object at frame {}'.format(frame))
    # iterate over each group

    for f, df_group in enumerate(df_grouped):
        if f==0:
            df_group=df_grouped[0]
            frame_p = df_group['frame'].values[0]
            df_gt_p = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
            print('First detected object at frame {}'.format(frame))
            continue

        frame = df_group['frame'].values[0]
        df_gt = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
        print(group_name)
        #bbox_iou(bboxA, bboxB)


    # Read BBox from 1st Frame
    #Bbox_picked = ut.non_max_suppression(bboxes, overlap_thresh)

if __name__ == '__main__':
    main()
