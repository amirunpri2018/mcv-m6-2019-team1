# -*- coding: utf-8 -*-
from __future__ import division
# Built-in modules
import glob
import logging
import os
import shutil
import time

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local modules
import src.utils as u


# Logger setup

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Directory in the root directory where the results will be saved
# Useful directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_DIR_GT = os.path.join(TRAIN_DIR, 'gt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

IMG_SHAPE = (1080, 1920)
threshold = 0.5 # IoU Threshold


if __name__ == '__main__':
    # List of DataFrames
    df = list()
    for d in os.listdir(os.path.join(ROOT_DIR, 'annotation_pascal')):
        # Get XML files from directory
        files = u.get_files_from_dir(
            os.path.join(ROOT_DIR, 'annotation_pascal', d),
            excl_ext='txt')
        # Obtain DataFrame from files
        df.append(u.get_bboxes_from_pascal(files, d))
    df = pd.concat(df)

    df_grouped = df.groupby('frame')

    vals = list()
    # iterate over each group
    for group_name, df_group in df_grouped:
        frame = df_group['frame'].values[0]
        #df_gt = df_group[['xmin', 'xmax', 'ymin', 'ymax']].values.tolist()
        #df_gt = df_group[['ymax', 'xmin', 'ymin', 'xmax']].values.tolist()
        #2nd attempt:
        df_gt = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()

        # Correct order: tly, tlx, bry, brx

        frame_vals = [frame]

        """
        1. Without noise: we expect excellent results
        """
        #print(df_group)
        bboxes1, fscore1, iou1, map1 = u.compute_metrics(df_gt, IMG_SHAPE,
                                                   noise_size=False,
                                                   noise_position=False,
                                                   create_bbox_proba=0,
                                                   destroy_bbox_proba=0,
                                                   k=5,
                                                   iou_thresh=threshold)
        frame_vals.extend([fscore1, iou1, map1])

        """
        2. Add noise: (noise param = 50)
        """

        bboxes2, fscore2, iou2, map2 = u.compute_metrics(df_gt, IMG_SHAPE,
                                                         noise_size_factor=5,
                                                         noise_position_factor=5,
                                                         create_bbox_proba=0,
                                                         destroy_bbox_proba=0,
                                                         k=5,
                                                         iou_thresh=threshold)
        frame_vals.extend([fscore2, iou2, map2])

        """
        3. Increase noise: (noise param = 100)
        """

        bboxes3, fscore3, iou3, map3 = u.compute_metrics(df_gt, IMG_SHAPE,
                                                         noise_size=10,
                                                         noise_position=10,
                                                         create_bbox_proba=0,
                                                         destroy_bbox_proba=0,
                                                         k=5,
                                                         iou_thresh=threshold)
        frame_vals.extend([fscore3, iou3, map3])

        """
        4. Add random Bboxes: (noise param = 100, bbox proba = 0.5)
        """

        bboxes4, fscore4, iou4, map4 = u.compute_metrics(df_gt, IMG_SHAPE,
                                                         noise_size=10,
                                                         noise_position=10,
                                                         create_bbox_proba=0.5,
                                                         destroy_bbox_proba=0,
                                                         k=5,
                                                         iou_thresh=threshold)
        frame_vals.extend([fscore4, iou4, map4])

        """
        5. Increase random Bboxes: (noise param = 100, bbox proba = 0.8)
        """

        bboxes5, fscore5, iou5, map5 = u.compute_metrics(df_gt, IMG_SHAPE,
                                                         noise_size=100,
                                                         noise_position=100,
                                                         create_bbox_proba=0.8,
                                                         destroy_bbox_proba=0,
                                                         k=5,
                                                         iou_thresh=threshold)
        frame_vals.extend([fscore5, iou5, map5])

        vals.append(frame_vals)

    df_metrics = pd.DataFrame(
        vals,
        columns=['frame', 'fscore1', 'iou1', 'map1', 'fscore2', 'iou2', 'map2', 'fscore3', 'iou3', 'map3',
                 'fscore4', 'iou4', 'map4', 'fscore5', 'iou5', 'map5']
    )

    frame_number = df_metrics['frame'].values.tolist()
    fscore1 = df_metrics['fscore1'].values.tolist()
    iou1 = df_metrics['iou1'].values.tolist()
    map1 = df_metrics['map1'].values.tolist()
    fscore2 = df_metrics['fscore2'].values.tolist()
    iou2 = df_metrics['iou2'].values.tolist()
    map2 = df_metrics['map2'].values.tolist()
    fscore3 = df_metrics['fscore3'].values.tolist()
    iou3 = df_metrics['iou3'].values.tolist()
    map3 = df_metrics['map3'].values.tolist()
    fscore4 = df_metrics['fscore4'].values.tolist()
    iou4 = df_metrics['iou4'].values.tolist()
    map4 = df_metrics['map4'].values.tolist()
    fscore5 = df_metrics['fscore5'].values.tolist()
    iou5 = df_metrics['iou5'].values.tolist()
    map5 = df_metrics['map5'].values.tolist()

    """
    TASK 2: Plot metrics against frame number
    """
    # TODO: Define frame_number!

    """
    Plot F-score
    """

    plt.figure()
    plt.plot(frame_number, fscore1, '-o', label = 'Original')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('F-score')
    plt.title('F-score in time')
    #plt.savefig(os.path.join('/home/agus/repos/mcv-m6-2019-team1/figures/orig_fscore_099.png'))



    plt.figure()
    #plt.plot(frame_number, fscore1, '-o', label = 'Original')
    #plt.plot(frame_number, fscore2, '-o', label = 'Noise 5px')
    #plt.plot(frame_number, fscore3, '-o', label = 'Noise 10px')
    #plt.plot(frame_number, fscore4, '-o', label = 'Noise 10px + 50% bboxes')
    #plt.plot(frame_number, fscore5, '-o', label = 'Noise 10px + 80% bboxes')
    plt.scatter(frame_number, fscore1, label='Original')
    plt.scatter(frame_number, fscore2, label='Noise 5px')
    plt.scatter(frame_number, fscore3, label='Noise 10px')
    plt.scatter(frame_number, fscore4, label='Noise 10px + 50% bboxes')
    plt.scatter(frame_number, fscore5, label='Noise 10px + 80% bboxes')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('F-score')
    plt.title('F-score in time')
    #plt.savefig(os.path.join(FIGURES_DIR, 'fscore.png'))
    plt.savefig(os.path.join('/home/agus/repos/mcv-m6-2019-team1/figures/scatter_fscore.png'))

    """
    Plot IoU
    """

    plt.figure()
    plt.plot(frame_number, iou1, '-o', label = 'Original')
    plt.plot(frame_number, iou2, '-o', label = 'Noise 5px')
    plt.plot(frame_number, iou3, '-o', label = 'Noise 10px')
    plt.plot(frame_number, iou4, '-o', label = 'Noise 10px + 50% bboxes')
    plt.plot(frame_number, iou5, '-o', label = 'Noise 10px + 80% bboxes')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    plt.title('IoU in time')
    #plt.savefig(os.path.join(FIGURES_DIR, 'iou.png'))
    plt.savefig(os.path.join('/home/agus/repos/mcv-m6-2019-team1/figures/iou.png'))


    """
    Plot Mapk:
    """

    plt.figure()
    plt.plot(frame_number, map1, '-o', label = 'Original')
    plt.plot(frame_number, map2, '-o', label = 'Noise 5px')
    plt.plot(frame_number, map3, '-o', label = 'Noise 10px')
    plt.plot(frame_number, map4, '-o', label = 'Noise 10px + 50% bboxes')
    plt.plot(frame_number, map5, '-o', label = 'Noise 10px + 80% bboxes')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('mAP')
    plt.title('mAP at 10 in time')
    #plt.savefig(os.path.join(FIGURES_DIR, 'map.png'))
    plt.savefig(os.path.join('/home/agus/repos/mcv-m6-2019-team1/figures/map.png'))



    """
    TASK 3: Optical Flow
    """

    # Compute Optical Flow of the two Kitti sequences
    # Obtain error metrics:
    # MSEN: Mean Square Error in Non-occluded areas
    # Use the following function:
    # optical_flow_mse = u.mse(gt_optical_flow, result_optical_flow)
    # PEPN: Percentage of Erroneous Pixels in Non-occluded areas
    # Draw histograms of resulting metrics:
    # hist = u.histogram(im_array, bins=128)

    """
    TASK 4: PLOT Optical Flow
    """

    # conf_mat = confusion_matrix(RESULT_DIR, TRAIN_MASKS_DIR)
    # print_confusion_matrix(conf_mat)
    # metrics = performance_evaluation_pixel(*conf_mat)
    # print_metrics(metrics)

"""
    # OLD WAY TO COMPUTE IT!!

    # Add noise to GT depending on noise parameter
    bboxes = u.add_noise_to_bboxes(df_gt, IMG_SHAPE,
                                   noise_size=True,
                                   noise_size_factor=5.0,
                                   noise_position=True,
                                   noise_position_factor=5.0)

    # Randomly create and destroy bounding boxes depending
    # on probability parameter
    bboxes = u.create_bboxes(bboxes, IMG_SHAPE, prob=0.5)
    bboxes = u.destroy_bboxes(bboxes, prob=0.5)


    Obtain TP, FP, TN metrics


    #ToDo: Vary 'bbox candidates' between the original gt, noisy gt, and randmomly created bboxes.

    window_candidates = df_gt # Original case
    window_candidates = bboxes # Noisy case

    [bboxTP, bboxFN, bboxFP] = performance_accumulation_window(window_candidates, df_gt)


    Compute F-score of GT against modified bboxes PER FRAME NUMBER

    # ToDo: Add dependency on frame number
    # ToDo: Create function fscore

    # FIRST CASE:
    #   GT against GT (we expect excellent results)
    fscore_original = list()

    for b, box in enumerate(df_gt):
        fscore_original.append(u.fscore(df_gt[b], df_gt[b]))

    # SECOND CASE:
    #   GT against noisy bboxes (we know the correspondance of GT against bbox)
    fscore_noisy = list()

    for b, box in enumerate(df_gt):
        fscore_noisy.append(u.fscore(df_gt[b], bboxes[b]))

    # THIRD CASE:
    #   GT against destroyed / added bboxes (we DON'T know the
    #   correspondance of GT against bbox)


    Compute IoU of GT against modified Bboxes PER FRAME NUMBER:

    # TODO: Add dependency on frame number
    # FIRST CASE:
    #   GT against GT (we expect excellent result)
    iou_original = list()

    for b, box in enumerate(df_gt):
        iou_original.append(u.bbox_iou(df_gt[b], df_gt[b]))

    # SECOND CASE:
    #   GT against noisy bboxes (we know the correspondance of GT against bbox)
    iou_noisy = list()

    for b, box in enumerate(df_gt):
        iou_noisy.append(u.bbox_iou(df_gt[b], bboxes[b]))

    # THIRD CASE:
    #   GT against destroyed / added bboxes (we DON'T know the
    #   correspondance of GT against bbox)
    # TODO


    Compute mAP of GT against modified bboxes PER FRAME NUMBER:


    # TODO: Add dependency on frame number
    # FIRST CASE:
    #   GT against GT (we expect excellent result)
    map_original = list()

    for b, box in enumerate(df_gt):
        map_original.append(u.mapk(df_gt[b], df_gt[b]))

    # SECOND CASE:
    #   GT against noisy bboxes (we know the correspondance of GT against bbox)
    map_noisy = list()

    for b, box in enumerate(df_gt):
        map_noisy.append(u.mapk(df_gt[b], bboxes[b]))

    # THIRD CASE:
    #   GT against destroyed / added bboxes (we DON'T know the
    #   correspondance of GT against bbox)
    # TODO

    """
