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
import utilsBG as uBG
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
threshold = 0.5 # IoU Threshold


if __name__ == '__main__':
    # List of DataFrames

    df = u.get_bboxes_from_MOTChallenge(TRAIN_DIR_GT)   # CHECK IF THIS IS THE CORRECT FUCTION!
    df_grouped = df.groupby('frame')

    vals = list()
    # iterate over each group

    for group_name, df_group in df_grouped:
        frame = df_group['frame'].values[0]
        df_gt = df_group[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()

        # Correct order: tly, tlx, bry, brx

        frame_vals = [frame]


        fscore, iou, map = uBG.compute_metrics_general(df_gt, BBOX!!!
                                                   k=5,
                                                   iou_thresh=0.5)
        frame_vals.extend([fscore, iou, map])

        vals.append(frame_vals)

    df_metrics = pd.DataFrame(
        vals,
        columns=['frame', 'fscore', 'iou', 'map']
    )

    frame_number = df_metrics['frame'].values.tolist()
    fscore = df_metrics['fscore'].values.tolist()
    iou = df_metrics['iou'].values.tolist()
    map = df_metrics['map'].values.tolist()

