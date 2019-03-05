# -*- coding: utf-8 -*-

# Built-in modules
import glob
import logging
import os
import shutil
import time

# 3rd party modules
import matplotlib as plt
import numpy as np

# Local modules
import utils as u


# Logger setup
from code.evaluation.evaluation_funcs import performance_accumulation_window

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


if __name__ == '__main__':
    # Get XML list from directory
    xml_gt = u.get_files_from_dir(TRAIN_DIR_GT)

    # Get GT from XML
    df_gt = u.get_bboxes_from_aicity(xml_gt)

    # NEW WAY TO COMPUTE IT!!

    bboxes, fscore, iou, map = u.compute_metrics(df_gt, IMG_SHAPE,
                                               noise_size=5,
                                               noise_position=5,
                                               create_bbox_proba=0.5,
                                               destroy_bbox_proba=0.5,
                                               k=10)
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

    """
    Obtain TP, FP, TN metrics
    """

    #ToDo: Vary 'bbox candidates' between the original gt, noisy gt, and randmomly created bboxes.

    window_candidates = df_gt # Original case
    window_candidates = bboxes # Noisy case

    [bboxTP, bboxFN, bboxFP] = performance_accumulation_window(window_candidates, df_gt)

    """
    Compute F-score of GT against modified bboxes PER FRAME NUMBER
    """
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

    """
    Compute IoU of GT against modified Bboxes PER FRAME NUMBER:
    """
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

    """
    Compute mAP of GT against modified bboxes PER FRAME NUMBER:
    """

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
    TASK 2: Plot metrics against frame number
    """
    # TODO: Define frame_number!
    # TODO: We can vary noise parameter and plot several noisy results in
    #   same graph, labeling them by noise param.

    frame_number = 1

    # Plot F-score
    plt.plot(fscore_original, '-o', frame_number, label = 'Original')
    plt.plot(fscore_noisy, '-o', frame_number, label='Noisy')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('F-score')
    plt.title('F-score in time')
    plt.savefig(os.path.join(FIGURES_DIR, 'fscore.png'))

    # Plot IoU
    plt.plot(iou_original, '-o', frame_number, label='Original')
    plt.plot(iou_noisy, '-o', frame_number, label='Noisy')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    plt.title('IoU in time')
    plt.savefig(os.path.join(FIGURES_DIR, 'iou.png'))

    # Plot Mapk :
    plt.plot(map_original, '-o', frame_number, label='Original')
    plt.plot(map_noisy, '-o', frame_number, label='Noisy')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('MaP')
    plt.title('MaP in time')
    plt.savefig(os.path.join(FIGURES_DIR, 'map.png'))

    # Should we plot the results against the noise parameter???
    # I think this makes more sense!!

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
