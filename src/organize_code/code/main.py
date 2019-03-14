# -*- coding: utf-8 -*-

# Built-in modules
import glob
import logging
import os
import shutil
import time

# Optical flow
import cv2 as cv

import utils_opticalFlow as ut
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')


# 3rd party modules
import matplotlib as plt
import numpy as np

# Local modules
import utils as u


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


if __name__ == '__main__':
    # Get XML list from directory
    xml_gt = u.get_files_from_dir(TRAIN_DIR_GT)

    # Get GT from XML
    df_gt = u.get_bboxes_from_aicity(xml_gt)

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

    [bboxTP, bboxFN, bboxFP] = evalf.performance_accumulation_window(window_candidates, df_gt)

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
    # Data Folders
    img_folder = '../datasets/kitti_optical_flow/training_sequences/'
    gt_folder = '../datasets/kitti_optical_flow/gt/'
    result_folder = '../datasets/kitti_optical_flow/kitti_LK_results/'
    # Sequance 1
    seq1 = '000045_10.png'
    seq1_2 = '000045_11.png'

    # Sequance 2
    seq2=  '000157_10.png'
    seq2_2 = '000157_11.png'
    # TASK 3
    # Read Ground Truth
    [u1gt,v1gt,valid1gt] = ut.readOF(gt_folder,seq1)
    [u2gt,v2gt,valid2gt] = ut.readOF(gt_folder,seq2)
    # Read LK Result
    [u1lk,v1lk,valid1lk] = ut.readOF(result_folder,'LKflow_'+seq1)
    [u2lk,v2lk,valid2lk] = ut.readOF(result_folder,'LKflow_'+seq2)
    task3_1 = True
    # TASK 3.1
    if task3_1:
    # Computing the MSE + PEPE
        # seq 1
        errmap1, mse1,pepe1 = ut.MSEN_PEPN(u1gt,v1gt,valid1gt,u1lk,v1lk,valid1lk)
        errmag1, errang1 = ut.err_flow(u1gt,v1gt,valid1gt,u1lk,v1lk,valid1lk)
        print("Seq 1")
        print("MSE:")
        print(mse1)
        print("pepe:")
        print(pepe1)
        print('-----------------------')

        # "Err map seq #1" ,
        # seq 2
        errmap2, mse2,pepe2 = ut.MSEN_PEPN(u2gt,v2gt,valid2gt,u2lk,v2lk,valid2lk)
        errmag2, errang2 = ut.err_flow(u2gt,v2gt,valid2gt,u2lk,v2lk,valid2lk)
        print("Seq 2")
        print("MSE:")
        print(mse2)
        print("pepe:")
        print(pepe2)
        print('-----------------------')


    # Visualization of Error
    Task_3_2 = True
    if Task_3_2:
        # Seq 1
        ut.OF_err_disp(errmap1,valid1gt,seq1)
        ut.OF_err_disp(errmag1,valid1gt,'Magnitude_'+seq1)
        ut.OF_err_disp(errang1,valid1gt,'Angle_' +seq1)
        # Seq 2
        ut.OF_err_disp(errmap2,valid2gt,seq2)
        ut.OF_err_disp(errmag2,valid2gt,'Magnitude_'+seq2)
        ut.OF_err_disp(errang2,valid2gt,'Angle_' +seq2)

    """
    TASK 4: PLOT Optical Flow
    """

    # conf_mat = confusion_matrix(RESULT_DIR, TRAIN_MASKS_DIR)
    # print_confusion_matrix(conf_mat)
    # metrics = performance_evaluation_pixel(*conf_mat)
    # print_metrics(metrics)
    Task_4 = False
    # TASK 3.4
    if Task_4:
    # There is a big diffrence between the 2 sequances - checking if it due to bias value or angle difference
    # read original images
        Im1 = cv.imread(img_folder+seq1,cv.IMREAD_GRAYSCALE)
        Im1_2 = cv.imread(img_folder+seq1_2,cv.IMREAD_GRAYSCALE)
        ut.plotOF(Im1,Im1_2, u1gt,v1gt,20,'Seq_GT_1')

        Im2 = cv.imread(img_folder+seq2,cv.IMREAD_GRAYSCALE)
        Im2_2 = cv.imread(img_folder+seq2_2,cv.IMREAD_GRAYSCALE)
        ut.plotOF(Im2, Im2_2,u2gt,v2gt,20,'Seq_GT_2')


        ut.plotOF(Im1,Im1_2, u1lk,v1lk,20,'Seq_LK_1')
        ut.plotOF(Im2, Im2_2,u2lk,v2lk,20,'Seq_LK_2')
