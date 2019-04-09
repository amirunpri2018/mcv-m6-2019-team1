# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib
from IPython import display as dp

# Related 3rd party libraries
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import io
import os
import time

# Local libraries
import utils.utils_kalman as ut
import organize_code.code.utils as code_ut
import utils.utils_xml as ut_xml
from utils.sort import Sort
import utils.map_metrics as mp

#import src

OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'

# Some constant for the script
TASK = 'task_kalman_multi'
WEEK = 'week5'

SEQ = 'S03'
CAM = 'c015'
INPUT = 'det_mask_rcnn'

CREATE_GT_PKL = True

# Set useful directories
frames_dir = os.path.join(
    ROOT_DIR,
    'train',
    SEQ,
    CAM,
    'frames')


results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK)

# Create folders if they don't exist
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


tstamp_path = os.path.join(ROOT_DIR, 'cam_timestamp', SEQ + '.txt')
time_offset, fps = ut.obtain_timeoff_fps(tstamp_path, CAM)

# Tracker graphic options:
display = False
save = False

def main():

    # Ground truth file path:

    gt_det = os.path.join(ROOT_DIR, 'train', SEQ, CAM, 'det', INPUT + '.txt')

    # Save gt file as pkl:

    if CREATE_GT_PKL == True:
        gt_txt = os.path.join(ROOT_DIR, 'train', SEQ, CAM, 'gt', 'gt.txt')
        save_gt_pkl = os.path.join(ROOT_DIR, 'train', SEQ, CAM, 'gt', 'gt.pkl')
        code_ut.getBBox_from_gt(gt_txt, save_in=save_gt_pkl)

    # Get BBox detection from list
    df = ut.get_bboxes_from_MOTChallenge(gt_det)
    df.loc[:, 'time_stamp'] = df['frame'].values.tolist()

    # Display data:
    colours = np.random.rand(32,3) # Used only for display

    # Sort and group bbox by frame:
    df.sort_values(by=['frame'])
    df_grouped = df.groupby('frame')

    total_time = 0.0
    total_frames = 0
    out = []

    if display:
        plt.ion()  # for iterative display
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    # Create instance of the SORT tracker
    mot_tracker = Sort()

    for f, df_group in df_grouped:

        frame = int(df_group['frame'].values[0])

        time_stamp = ut.timestamp_calc(frame, time_offset, fps)

        df_gt = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()


        # Reshape GT format for Kalman tracking algorithm:
        # [x1,y1,x2,y2] format for the tracker input:

        df_gt = np.asarray(df_gt)
        dets = np.stack([df_gt[:,1], df_gt[:,0], df_gt[:,3], df_gt[:,2]], axis=1)
        dets = np.reshape(dets, (len(dets), -1))
        dets = np.asarray(dets, dtype=np.float64, order='C')


        if (display):
            fn = frames_dir + '/frame_%04d.jpg' % (frame)  # read the frame
            im = io.imread(fn)
            ax[0].imshow(im)
            ax[0].axis('off')
            ax[0].set_title('Original R-CNN detections (untracked)')
            for j in range(np.shape(dets)[0]):
                color = 'r'
                coords = (dets[j, 0].astype(np.float), dets[j, 1].astype(np.float)), dets[j, 2].astype(np.float) - dets[j, 0].astype(np.float), dets[j, 3].astype(np.float) - dets[j, 1].astype(np.float)
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

        out.append([frame, trackers, time_stamp])

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

    ############################################################################################################

    # Result of Kalman tracking (pandas format):

    df_kalman = ut.kalman_out_to_pandas_for_map(out)

    # Save kalman filter output:

    ut.save_pkl(df_kalman, os.path.join(results_dir, INPUT + SEQ + CAM +'_kalman_predictions.pkl'))

    #df_pred_corr = ut.panda_to_json_predicted(df_kalman)
    #ut.save_json(df_pred_corr, os.path.join(results_dir, INPUT + '_predicted_boxes.json'))


