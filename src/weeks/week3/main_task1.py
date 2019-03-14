# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local libraries
import src


# Some constant for the script
N = 0.01
GT = 'no'
DIM = 3
EXP_NAME = '{}GT_N{}_DIM{}'.format(GT, N, DIM)
TASK = 'task2'
WEEK = 'week2'


def main():
    """
    Add documentation.

    :return: Nothing
    """

    # Set useful directories
    frames_dir = os.path.join(
        src.ROOT_DIR,
        'datasets',
        'm6_week1_frames',
        'frames')
    results_dir = os.path.join(src.OUTPUT_DIR, WEEK, TASK, EXP_NAME)
    # Ground truth file path
    gt_file = os.path.join(src.ROOT_DIR,
                           'datasets', 'AICity_data', 'train', 'S03',
                           'c010', 'gt', 'gt.txt')

    # Create folders if they don't exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)


if __name__ == '__main__':
    main()
