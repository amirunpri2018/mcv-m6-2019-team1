
import numpy as np
import cv2 as cv

import utils_opticalFlow as ut
#import matplotlib
#matplotlib.use('Agg')



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

#Im1 = cv.imread(result_folder+'LKflow_'+seq1,-1)


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

Task_3_4 = False
# TASK 3.4
if Task_3_4:
# There is a big diffrence between the 2 sequances - checking if it due to bias value or angle difference
# read original images
    Im1 = cv.imread(img_folder+seq1,cv.IMREAD_GRAYSCALE)
    Im1_2 = cv.imread(img_folder+seq1_2,cv.IMREAD_GRAYSCALE)
    ut.plotOF(Im1,Im1_2,u1gt,v1gt, 10,'Seq_GT_1')

    Im2 = cv.imread(img_folder+seq2,cv.IMREAD_GRAYSCALE)
    Im2_2 = cv.imread(img_folder+seq2_2,cv.IMREAD_GRAYSCALE)
    ut.plotOF(Im2, Im2_2,u2gt,v2gt,10,'Seq_GT_2')


    ut.plotOF(Im1,Im1_2, u1lk,v1lk,10,'Seq_LK_1')
    ut.plotOF(Im2, Im2_2,u2lk,v2lk,10,'Seq_LK_2')
