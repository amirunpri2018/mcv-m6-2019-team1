# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib
# matplotlib.rcParams['backend'] = 'TkAgg'

# matplotlib.use('TkAgg')
from IPython import display as dp

import numpy as np
from skimage import io
import os
import time

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local libraries
import utils as ut
from sort import Sort

import map_metrics as mp




##################################################################

import pandas as pd


def locate_panda_feature(pd_dataframe, key, value):
    """
    Group pandas dataframe by selected key
    """
    grouped_pd = pd_dataframe.groupby(key).agg(lambda x: list(x))

    return grouped_pd.loc[value]


# locate_panda_feature(df_kalman, 'img_id', 425)

#####################################################################

def kalman_out_to_pandas(out_kalman):
    """
    :param out_kalman: Output from kalman tracking
    :returns: Panda dataframe with format 'frame', 'ymin', 'xmin', 'ymax', 'xmax', 'track_id'
    """

    vals = list()

    for frame_data in out_kalman:

        frame = frame_data[0]
        frame_vals = [frame]

        for track in frame_data[1]:
            ymin, xmin, ymax, xmax, track_id = track

            #score = 1
            scores = np.random.uniform(0,1)

            frame_vals = [frame, ymin, xmin, ymax, xmax, track_id, scores]

            vals.append(frame_vals)

    df_kalman = pd.DataFrame(vals, columns=['frame', 'ymin', 'xmin', 'ymax', 'xmax', 'track_id', 'scores'])

    return df_kalman


###################################################

def kalman_out_to_pandas_for_map(out_kalman):
    """
    Prepair dictionary for map_metrics.py
    :param out_kalman: Output from kalman tracking
    :returns: Panda dataframe with format 'img_id', 'boxes', 'track_id', 'scores'
    """

    vals = list()

    for frame_data in out_kalman:

        img_id = frame_data[0]
        frame_vals = [frame]

        for track in frame_data[1]:
            ymin, xmin, ymax, xmax, track_id = track

            boxes = [xmin, ymin, xmax, ymax]

            #scores = 1
            scores = np.random.uniform(0, 1)

            frame_vals = [img_id, boxes, track_id, scores]

            vals.append(frame_vals)

    df_kalman = pd.DataFrame(vals, columns=['img_id', 'boxes', 'track_id', 'scores'])

    return df_kalman


###########################################################

def get_bboxes_from_MOTChallenge_for_map(fname):
    """
    Read GT as format required in map_metrics.py

    Get the Bboxes from the txt files
    MOTChallengr format [frame,ID,left,top,width,height,1,-1,-1,-1]
     {'ymax': 84.0, 'frame': 90, 'track_id': 2, 'xmax': 672.0, 'xmin': 578.0, 'ymin': 43.0, 'occlusion': 1}
    fname: is the path to the txt file
    :returns: Pandas DataFrame with the data
    """
    f = open(fname, "r")
    BBox_list = list()

    for line in f:
        data = line.split(',')
        xmax = float(data[2]) + float(data[4])
        ymax = float(data[3]) + float(data[5])

        BBox_list.append({'img_id': int(data[0]),
                          'track_id': int(data[1]),
                          'boxes': [float(data[2]), float(data[3]), xmax, ymax],
                          'occlusion': 1,
                          'conf': float(data[6])})

    return pd.DataFrame(BBox_list)

# map metrics: xmin, ymin, xmax, ymax

##########################################################



# import src
OUTPUT_DIR = '../../../../output'
ROOT_DIR = '../../../../'

# Some constant for the script
N = 0.01
GT = 'no'
DIM = 3
EXP_NAME = '{}GT_N{}_DIM{}'.format(GT, N, DIM)
TASK = 'task_kalman'
WEEK = 'week3'

# Set useful directories
frames_dir = os.path.join(
    ROOT_DIR,
    'data',
    'm6_week1_frames',
    'frames')
results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK)

# Create folders if they don't exist
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

# Ground truth file path
gt_file = os.path.join(ROOT_DIR,
                       'data', 'AICity_data', 'train', 'S03',
                       'c010', 'gt', 'gt.txt')

# Get BBox detection from list
df = ut.get_bboxes_from_MOTChallenge(gt_file)

# Change track id for testing algorithm:
df.loc[:, 'track_id'] = -1

df.sort_values(by=['frame'])

# Group bbox by frame
df_grouped = df.groupby('frame')

vals = list()

# Display data:

colours = np.random.rand(32, 3)  # Used only for display

# Run tracker:

display = False
save = False
total_time = 0.0
total_frames = 0
out = []

if display:
    plt.ion()  # for iterative display
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

mot_tracker = Sort()  # create instance of the SORT tracker

for f, df_group in df_grouped:
    if f == 0:
        df_group = df_grouped[0]
        frame_p = df_group['frame'].values[0]
        df_gt_p = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
        print('First detected object at frame {}'.format(frame))
        continue

    frame = df_group['frame'].values[0]
    df_gt = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
    df_gt = np.asarray(df_gt)

    dets = np.stack([df_gt[:, 1], df_gt[:, 0], df_gt[:, 3], df_gt[:, 2]], axis=1)

    dets = np.reshape(dets, (len(dets), -1))

    if (display):
        fn = '../../../../frames/frame_%03d.jpg' % (frame)  # read the frame
        im = io.imread(fn)

        video_name = 'video_kalman.avi'
        height, width, layers = im.shape
        video = cv.VideoWriter(video_name, 0, 1, (width, height))

        ax[0].imshow(im)
        ax[0].axis('off')
        ax[0].set_title('Original Faster R-CNN detections')
        for j in range(np.shape(dets)[0]):
            color = colours[j]
            coords = (dets[j, 0], dets[j, 1]), dets[j, 2] - dets[j, 0], dets[j, 3] - dets[j, 1]
            ax[0].add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, lw=3))

    total_frames += 1

    if (display):
        ax[1].imshow(im)
        ax[1].axis('off')
        ax[1].set_title('Tracked Targets')

    start_time = time.time()
    trackers = mot_tracker.update(dets)
    cycle_time = time.time() - start_time
    total_time += cycle_time

    out.append([frame, trackers])

    for d in trackers:
        if (display):
            d = d.astype(np.uint32)
            ax[1].add_patch(
                patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
            ax[1].set_adjustable('box-forced')

    if (save):
        plt.savefig(os.path.join(results_dir, 'video_kalman_' + str(frame) + '.png'))

    if (display):
        dp.clear_output(wait=True)
        dp.display(plt.gcf())
        time.sleep(0.000001)
        ax[0].cla()
        ax[1].cla()

plt.show()

print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

#df_kalman = kalman_out_to_pandas(out)

#dict_kalman = pd.DataFrame.to_dict(df_kalman)

df_kalman = kalman_out_to_pandas_for_map(out)

df_gt = get_bboxes_from_MOTChallenge_for_map(gt_file)

map_metric = mp.get_avg_precision_at_iou(df_gt, df_kalman, iou_thr=0.5)

