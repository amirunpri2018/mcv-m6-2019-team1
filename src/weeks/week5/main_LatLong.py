# -*- coding: utf-8 -*-

# Standard libraries
import os
# import List libariry
import pandas as pd
import numpy as  np
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut

def addlatlong(df,H):
    df.loc[:,'Lat'] = 0.0
    df.loc[:,'Long'] = 0.0
    #df.loc[:,'Alt'] = 0.0
    for i in df.index.tolist():
        x = df.loc[i,'Mx']
        y = df.loc[i,'My']
        pxgps = np.dot(H,np.array([y,x,1.0]))
        pxgps = pxgps / pxgps[-1]
        df.loc[i,'Lat'] = pxgps[0]
        df.loc[i,'Long'] = pxgps[1]

    return df

#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'
# Some constant for the script
# Some constant for the script

TASK = 'SC_YOLO_KALMAN_LLA'
WEEK = 'week5'
def main():
    """
    Add documentation.

    :return: Nothing
    """

    # Set useful directories

    EXP_LIST = os.listdir(os.path.join(OUTPUT_DIR,WEEK,TASK))
    #EXP_LIST = [EXP_LIST[2]]
    #print(EXP_LIST)
    #EXP_LIST = ['S03_c015_RCNN_KALMAN_N1','S03_c011_RCNN_KALMAN_N1']
    print('Adding Lat Long approximation to Bbox detections')
    for EXP_NAME in EXP_LIST:

        print(EXP_NAME)
        print('------------------')
        results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)

        if not os.path.isdir(results_dir):
            continue

        fname = os.path.join(results_dir,"pred_tracks.pkl")
        if os.path.isfile(fname):
            # ger camera number:
            SEQ = EXP_NAME.split('_')[0]
            CAM = EXP_NAME.split('_')[1]
            # get camera calibration iou_matrix
            print('{} calibration..'.format(CAM))
            cam_cal_file = os.path.join(os.path.join(ROOT_DIR, 'train', SEQ, CAM),'calibration.txt')

            cal_matrix = ut.get_cal_matrix(cam_cal_file)
            Hinv = np.linalg.inv(cal_matrix)
            df_track = ut.getBBox_from_gt(fname)
            if not 'Mx' in df_track.head(0):
                print('calc motion..')
                df_track = ut.get_trackMotion(df_track)
            df_track = addlatlong(df_track,Hinv)

            os.rename(fname, os.path.join(results_dir,'pred_tracks0.pkl'))
            df_track.to_pickle(fname)
            csv_file = os.path.splitext(fname)[0] + "_lla.csv"
            export_csv = df_track.to_csv(csv_file, sep='\t', encoding='utf-8')
            print('LLA was added..')
        else:
            print('No prediction file in {}'.format(EXP_NAME))
