# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
# Standard libraries
import os
# import List libariry
import pandas as pd
import numpy as  np
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut


#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'


TASK = 'SC_YOLO_KALMAN_LLA'
WEEK = 'week5'
def main():
    """
    Add documentation.

    :return: Nothing
    """

    # Set useful directories

    EXP_LIST = os.listdir(os.path.join(OUTPUT_DIR,WEEK,TASK))

    #get all the pd list
    cam_list = list()
    cam_dict = list()
    max_time = 0
    minLAT = 43.0
    maxLAT = 0
    minLONG = -90.0
    maxLONG = -100.0
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
            cam_dict.append(CAM)
            df_track = ut.getBBox_from_gt(fname)
            print(df_track.head(0))
            max_time = max(max_time,max(df_track['time_stamp'].tolist()))
            # to create the plot
            maxLAT = max(maxLAT,max(df_track['Lat'].tolist()))
            maxLONG = max(maxLONG,max(df_track['Long'].tolist()))
            minLAT = min(minLAT,min(df_track['Lat'].tolist()))
            minLONG = min(minLONG,min(df_track['Long'].tolist()))
            cam_list.append(df_track)



        else:
            print('No prediction file in {}'.format(fname))



    # Run on sliding window of N frames
    WIN_SIZE = 30.0 # [sec]
    OVERLAP = 0.5

    cam_marker = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    colors = 'bgrcmykw'

    #plt.ion()
    #plt.show()
    #fig = plt.figure()
    #fig.suptitle('Tracks Trajectories', fontsize = 16)

    #ax = fig.add_subplot(111)
    #ax.scatter(Lat, Long, marker='.',color = colors[cl_idx] );
    #ax.scatter([minLAT,maxLAT], [minLONG,maxLONG], marker='.',color = 'k' );
    for t_start in np.arange(0,max_time - WIN_SIZE,OVERLAP):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # look for all the Cameras with bbox with this time frame
        #current_win = list()
        for idx,c in enumerate(cam_list):
            current_win=c.loc[(c['time_stamp']>=t_start) & (c['time_stamp']<t_start+WIN_SIZE)]
            print(cam_dict[idx])
            trk_c = current_win.groupby('track_id')
            #ax.clear()
            #ax.autoscale(False)
            #ax.scatter([minLAT,maxLAT], [minLONG,maxLONG], marker='.',color = 'k' );

            for id, tk in trk_c:
                cl_idx = int(id) % 8
                Lat = tk['Lat']
                Long = tk['Long']
                #print([minLAT,maxLAT])
                #print([minLONG,maxLONG])
                ax.scatter(Lat, Long, marker=cam_marker[idx],color = colors[cl_idx] );


                #ax.grid(True)
                #plt.hold(True)
                #plt.show()
        ax.grid(True)
        ax.set_title('time {}-{}'.format(t_start,t_start+WIN_SIZE))
        #ax.set_xlim([minLAT,maxLAT])
        #ax.set_ylim([minLONG,maxLONG])
        fig.suptitle('Tracks Trajectories', fontsize = 16)
        plt.show()

        #matches = compare_gps_tracks(current_win)





        """
                    os.rename(fname, os.path.join(results_dir,'pred_tracks0.pkl'))
                    df_track.to_pickle(fname)
                    csv_file = os.path.splitext(fname)[0] + "_lla.csv"
                    export_csv = df_track.to_csv(csv_file, sep='\t', encoding='utf-8')
                    print('LLA was added..')
        """
