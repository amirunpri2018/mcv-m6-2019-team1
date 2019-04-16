# -*- coding: utf-8 -*-

# Standard libraries
import os
# import List libariry
import pandas as pd
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut

#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'
# Some constant for the script
# Some constant for the script

TASK = 'SC_YOLO_S01'
WEEK = 'week5'
def main():
    """
    Add documentation.

    :return: Nothing
    """

    # Set useful directories

    EXP_LIST = os.listdir(os.path.join(OUTPUT_DIR,WEEK,TASK))
    #EXP_LIST = ['S03_c015_RCNN_KALMAN_N1','S03_c011_RCNN_KALMAN_N1']

    for EXP_NAME in EXP_LIST:

        print(EXP_NAME)
        print('------------------')
        results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)

        if not os.path.isdir(results_dir):
            continue

        fname = os.path.join(results_dir,"pred_tracks.pkl")
        if os.path.isfile(fname):
            df_track = ut.getBBox_from_gt(fname)
            df_track = ut.track_cleanup(df_track,MIN_TRACK_LENGTH=10,STATIC_OBJECT = 100) # object moved less than 15 pix
            #os.rename(save_in, os.path.join(results_dir,'pred_tracks0.pkl'))
            df_track.to_pickle(os.path.join(results_dir,"pred_tracks.pkl"))
        else:
            print('No prediction file in {}'.format(EXP_NAME))
