# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# import List libariry
import pandas as pd
#from itertools import repeat
# Local libraries
import organize_code.code.utils as ut
import organize_code.code.utils_opticalFlow as of
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
# import pyopt
# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
# from __future__ import unicode_literals

from PIL import Image
import time
import argparse
#import pyflow
from pyflow import pyflow
# for Learning
from sklearn import linear_model
#import src
OUTPUT_DIR ='../output'
ROOT_DIR = '../'
# Some constant for the script
N = 2
DET = 'YOLO'
EXP_NAME = '{}_N{}'.format(DET, N)
TASK = 'task3'
WEEK = 'week4'
DET_GAP = 5
PLOT_FLAG = False
VID_FLAG = False
SAVE_FLAG = True
LEARN_FLAG = False
CLEAN_FLAG = False
TRACK_FLAG = True

def main():
    """
    Add documentation.

    :return: Nothing
    """


    # ONLY THE FALSE ALARMS and the TRACKING can be improved - but the missdetecion are lost - so the highest preciosion we can get has a ceiling
    #Initialize Track ID - unique ascending numbers

    # Set useful directories
    frames_dir = os.path.join(
        ROOT_DIR,
        'data',
        'm6_week1_frames',
        'frames')
    results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)

    # Create folders if they don't exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    # detection file can be txt/xml/pkl
    det_file = os.path.join(ROOT_DIR,
                           'data', 'AICity_data', 'train', 'S03',
                           'c010', 'det', 'det_yolo3.txt')
    # grond truth - used only for learning
    gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation.pkl')
    # Get BBox detection from list
    df = ut.getBBox_from_gt(det_file)

    df.sort_values(by=['frame'])

    df.loc[:,'track_id'] = -1
    # New columns for tracking

    df.loc[:,'track_iou_score'] = -1.0
    # Motion
    df.loc[:,'Dx'] = -300.0
    df.loc[:,'Dy'] = -300.0
    df.loc[:,'ofDu'] = -300.0
    df.loc[:,'ofDv'] = -300.0
    df.loc[:,'rot'] = -1.0
    df.loc[:,'zoom'] = -1.0
    df.loc[:,'Mx'] = -1.0
    df.loc[:,'My'] = -1.0
    df.loc[:,'area'] = -1.0
    df.loc[:,'ratio'] = -1.0
    df.loc[:,'OF_err'] = 10
    # Group bbox by frame
    df_grouped = df.groupby('frame')

    vals = list()

    # Get first bbox

    frame_p = 0
    df_gt_p = []

    # iterate over each group
    #df_track = pd.DataFrame({'frame':[]:'ymin':[], 'xmin':[], 'ymax':[], 'xmax':[]})
    headers = list(df.head(0))
    print(headers )
    print('---------')


    if LEARN_FLAG:
        #dfgt = ut.getBBox_from_gt(gt_file)
        dfgt = ut.getBBox_from_gt(det_file)
        dfgt.sort_values(by=['frame'])
        dfgt_grouped = dfgt.groupby('frame')
        dfgt_list = pd.DataFrame({'width':[],'height':[], 'numel':[], 'du_std':[], 'dv_std':[],'du':[], 'dv':[],'track_id':[]})
        print('Learning Linear Regression from GT...')
        for f, df_group in dfgt_grouped:

            df_group = df_group.reset_index(drop=True)
            res_file = os.path.join( results_dir,'regression.pkl')
            if f%20==0:
                print(f)
            if f>250:
                break
            if f<220:
                continue

            target_frame_path = os.path.join(frames_dir,'frame_'+str(int(f)).zfill(3)+'.jpg')
            anchor_frame_path = os.path.join(frames_dir,'frame_'+str(int(f)-1).zfill(3)+'.jpg')

            # Forward compenstation ----> can be changed if needed - just change the order

            im1 = np.array(Image.open(target_frame_path))
            im2 = np.array(Image.open(anchor_frame_path))
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.

            v, u, im2W = pyflow.coarse2fine_flow(
                    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                    nSORIterations, colType)

            bbox =  bb.bbox_list_from_pandas(df_group)
            #[ymin,xmin,ymax,xmax]

            for t in range(len(df_group)):
                c_bbox = bbox[t]
                box_motion = bb.getMotionBbox(df_group.ix[[t]],df_group.ix[[t]])
                xv, yv = np.meshgrid(range(int(c_bbox[1]),int(c_bbox[3])),range(int(c_bbox[0]),int(c_bbox[2])))

                c_u = u[ yv,xv]
                c_v = v[yv,xv]

                dfgt_tmp = pd.DataFrame({'width':[np.shape(c_v)[1]],'height':[np.shape(c_v)[0]], 'numel':[np.prod(np.shape(c_v))], 'du_std':c_u.std(), 'dv_std':c_v.std()})
                # 1 std is 68%, we assume the car cover at least 68% of the bbox area

                dfgt_list = dfgt_list.append(dfgt_tmp)

                #print(c_bbox)
                #distribution of OF


                #plot 1st frame
                if PLOT_FLAG:
                    mag, ang = cv.cartToPolar(c_u,c_v)
                    hsv = of.OF2hsv(c_u, c_v)
                    n_bins = 20
                    bins_u = np.linspace(np.min(mag), np.max(mag), num=n_bins)
                    bins_v = np.linspace(np.min(ang), np.max(ang), num=n_bins)
                    #val_hist, bin_centers = ut.histogram(np.asarray(Ap_perTrack), bins=len(Ap_perTrack)/10)
                    fig = plt.figure(1)

                    ax3 = plt.subplot(231)
                    ut.plot_bboxes(im1,[c_bbox],l=[t],ax=ax3,title = str(f))
                    #ax3.imshow(im1)
                    ax4 = plt.subplot(232)
                    ax4.imshow(im1[ yv,xv])

                    ax1 = plt.subplot(235)
                    ax1.hist(c_u.ravel(), bins=bins_u,alpha=0.65) #,width=0.8

                    ax1.set_title('magnitude hist \n std:{:.2f},mu:{:.2f}'.format(mag.std(),mag.mean()))
                    ax2 = plt.subplot(234)
                    ax2.hist(c_v.ravel(), bins=bins_v,alpha=0.65) #,width=0.8
                    ax2.set_title('angle hist \n std:{:.2f},mu:{:.2f}'.format(ang.std(),ang.mean()))
                    ax5 = plt.subplot(233)
                    ax5.imshow(hsv)
                    ax5.set_title('OF map')
                    plt.show()

        dfgt_list = dfgt_list.dropna()
        csv_file = os.path.splitext(res_file)[0] + ".csv"
        export_csv = dfgt_list.to_csv(csv_file, sep='\t', encoding='utf-8')
        dfgt_list.to_pickle(res_file)
        print('Linear regression...')
        reg = linear_model.LinearRegression()
        lm=reg.fit(dfgt_list, np.ones((len(dfgt_list),1)))
        lm.coef_
        print(lm.coef_)


    if CLEAN_FLAG:

        if PLOT_FLAG:
            a=1
            #plt.ion()
            #plt.show()
            #fig = plt.figure()
            #ax1 = fig.add_subplot(211)
            #ax2 = fig.add_subplot(212)

        # I. 1st step - removing bbox with un coherent motion -
        for f, df_group in df_grouped:

            df_group = df_group.reset_index(drop=True)
            if f%20==0:
                print(f)
            if f>250:
                break

            if f<215:
                continue

            target_frame_path = os.path.join(frames_dir,'frame_'+str(int(f)).zfill(3)+'.jpg')
            anchor_frame_path = os.path.join(frames_dir,'frame_'+str(int(f)+1).zfill(3)+'.jpg')

            # Forward compenstation ----> can be changed if needed - just change the order

            im1 = np.array(Image.open(target_frame_path))
            im2 = np.array(Image.open(anchor_frame_path))
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.



            u, v, im2W = pyflow.coarse2fine_flow(
                    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                    nSORIterations, colType)

            #flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            bbox =  bb.bbox_list_from_pandas(df_group)
            #[ymin,xmin,ymax,xmax]
            for t in range(len(df_group)):
                c_bbox = bbox[t]

                xv, yv = np.meshgrid(range(int(c_bbox[1]),int(c_bbox[3])),range(int(c_bbox[0]),int(c_bbox[2])))


                c_u = u[ yv,xv]
                c_v = v[yv,xv]
                mag, ang = cv.cartToPolar(c_u,c_v)
                #distribution of OF

                #plot 1st frame
                if PLOT_FLAG:
                    n_bins = 20
                    bins_u = np.linspace(np.min(mag), np.max(mag), num=n_bins)
                    bins_v = np.linspace(np.min(ang), np.max(ang), num=n_bins)
                    #val_hist, bin_centers = ut.histogram(np.asarray(Ap_perTrack), bins=len(Ap_perTrack)/10)
                    fig = plt.figure(1)

                    ax3 = plt.subplot(231)
                    ut.plot_bboxes(im1,[c_bbox],l=[t],ax=ax3,title = str(f))
                    #ax3.imshow(im1)
                    ax4 = plt.subplot(232)
                    ax4.imshow(im1[ yv,xv])

                    ax1 = plt.subplot(233)
                    ax1.hist(c_u.ravel(), bins=bins_u,alpha=0.65) #,width=0.8
                    ax1.set_title('magnitude hist')
                    ax2 = plt.subplot(234)
                    ax2.hist(c_v.ravel(), bins=bins_v,alpha=0.65) #,width=0.8
                    ax2.set_title('angle hist')
                    plt.show()
                #ax2.imshow(im1)
                #rgb = of.OF2hsv(u, v)
                #ax1.imshow(rgb)

    if TRACK_FLAG:
        print('Tracking IOU and PF')

        df_track = pd.DataFrame(columns=headers)

        #Initialize Track ID - unique ascending numbers
        Track_id = 8
        if PLOT_FLAG:

            plt.ion()
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax2 = fig.add_subplot(212)

        for f, df_group in df_grouped:
            df_group = df_group.reset_index(drop=True)
            if f%50==0:
                print(f)
            if f>300:
                break
            if f<200:
                continue

            im_path = os.path.join(frames_dir,'frame_'+str(int(f)).zfill(3)+'.jpg')

            # 1st frame -
            if frame_p ==0:

                frame_p = df_group['frame'].values[0]
                print('First detected object at frame {}'.format(frame_p))
                        # Assign new tracks
                for t in range(len(df_group)):

                    #print(df_group.loc['track_id'])
                    df_group.at[t, 'track_id'] = Track_id
                    Track_id+=1

                df_p_group = pd.DataFrame(columns=headers)
                df_p_group = df_p_group.append(df_group, ignore_index=True)
                df_p_group = df_p_group.dropna()
                Track_id +=len(df_group)

                df_track = df_track.append(df_p_group, ignore_index=True)
                continue




            frame_p = df_group['frame'].values[0]

            # if there is more than N frames between detection - it is a new track - even if it in the bbox overlaps

            if df_p_group['frame'].values[0]+DET_GAP >df_group['frame'].values[0]:


                target_frame_path = os.path.join(frames_dir,'frame_'+str(int(f)).zfill(3)+'.jpg')
                anchor_frame_path = os.path.join(frames_dir,'frame_'+str(int(f)-1).zfill(3)+'.jpg')

                # Forward compenstation ----> can be changed if needed - just change the order

                im1 = np.array(Image.open(target_frame_path))
                im2 = np.array(Image.open(anchor_frame_path))
                im1 = im1.astype(float) / 255.
                im2 = im2.astype(float) / 255.

                u, v, im2W = pyflow.coarse2fine_flow(
                        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                        nSORIterations, colType)

                bbox =  bb.bbox_list_from_pandas(df_group)
                prev_bbox =  bb.bbox_list_from_pandas(df_p_group)
                #[ymin,xmin,ymax,xmax]


                iou_mat = bb.bbox_lists_iou(df_p_group,df_group)
                #print('first',iou_mat)
                matchlist,iou_score = bb.match_iou(iou_mat,iou_th=0)

                # sort it according to the new frame
                offset = np.min(df_group.index.values.tolist() )
                #print(offset)
                if PLOT_FLAG:

                    #print(df_p_group['track_id'].tolist())
                    ut.plot_bboxes(im1,bbox,l=[df_group['track_id'].tolist(),df_group['track_iou_score'].tolist()],ax=ax,title = "frame: "+str(f))

                for t,iou_s in zip(matchlist,iou_score):
                    c_bbox = bbox[t[1]]
                    p_bbox = prev_bbox[t[0]]
                    du_b,dv_b = of.ArtificialOF(c_bbox,p_bbox)


                    xv, yv = np.meshgrid(range(int(c_bbox[1]),int(c_bbox[3])),range(int(c_bbox[0]),int(c_bbox[2])))
                    xvp, yvp = np.meshgrid(range(int(p_bbox[1]),int(p_bbox[3])),range(int(p_bbox[0]),int(p_bbox[2])))
                    c_u = u[yv,xv]
                    c_v = v[yv,xv]

                    valid = np.ones(np.shape(c_u), dtype=bool)




                    # Motion parameters
                    #------------------
                    # Dx ,Dv - of bbox center
                    # Zoom - Ratio of areas
                    # Rot - Ratio of Ratio(h/w)'' -describes rotation

                    box_motion = bb.getMotionBbox(df_p_group.ix[[t[0]]],df_group.ix[[offset+t[1]]])
                    err_mot = np.sqrt((np.mean(c_u)-box_motion[1])**2+(np.mean(c_v)-box_motion[0])**2)
                    err_map,msen,pepn = of.MSEN_PEPN(c_u, c_v, valid, du_b, dv_b, valid,ERR_TH =err_mot)
                    magof = np.sqrt((np.mean(c_u))**2+(np.mean(c_v))**2)
                    magd = np.sqrt((box_motion[1])**2+(box_motion[0])**2)
                    max_mot = np.max([magof,magd])
                    nerr_mot = err_mot/max_mot
                    # if it sub pixel - it doest matter
                    err_mot_of = err_mot/np.max([magof,1])
                    if err_mot_of<25:
                        if PLOT_FLAG:
                            print(df_p_group.get_value(t[0],'track_id'))
                            print(':')
                            print('msen:{}'.format(msen))
                            print('pepn:{}'.format(pepn))
                            print('err_mot:{}'.format(err_mot))
                            print('nerr_mot:{}'.format(nerr_mot))
                            print('magd:{}'.format(magd))
                            print('magof:{}'.format(magof))
                            print('err_mot_of:{}'.format(err_mot_of))
                            fig = plt.figure()
                            ax7 = fig.add_subplot(311)
                            cax7 = ax7.imshow(err_map)
                            cbar7 = fig.colorbar(cax7,orientation='horizontal')
                            std2 = np.sqrt(np.mean(np.abs(err_map - np.median(err_map))**2))
                            ax7.set_title('track id:{} \n err:{}'.format(df_p_group.get_value(t[0],'track_id'),err_mot_of))
                            ax8 = fig.add_subplot(312)
                            cax8 = ax8.imshow(im2[yvp,xvp])
                            ax8.set_title(p_bbox)
                            ax9 = fig.add_subplot(313)
                            cax9 = ax9.imshow(im1[yv,xv])
                            ax9.set_title(c_bbox)
                            plt.show()
                        df_group.at[offset+t[1], 'ofDu'] = c_u.mean()
                        df_group.at[offset+t[1], 'ofDv'] = c_v.mean()
                        df_group.at[offset+t[1], 'track_id'] = df_p_group.get_value(t[0],'track_id')
                        df_group.at[offset+t[1], 'track_iou_score'] = iou_s
                        df_group.loc[[offset+t[1]],'rot'] = box_motion[3]#.columns = ['Dy', 'Dx','zoom','rot','Mx','My']
                        df_group.loc[[offset+t[1]],'zoom'] = box_motion[2]
                        df_group.loc[[offset+t[1]],'Dx'] = box_motion[1]
                        df_group.loc[[offset+t[1]],'Dy'] = box_motion[0]
                        df_group.loc[[offset+t[1]],'Mx'] = box_motion[4]
                        df_group.loc[[offset+t[1]],'My'] = box_motion[5]
                        df_group.loc[[offset+t[1]],'area'] = box_motion[6]
                        df_group.loc[[offset+t[1]],'ratio'] = box_motion[7]
                        df_group.loc[[offset+t[1]],'OF_err'] = err_mot_of

                    else:
                        print('x')
                        print('err_mot_of:{}'.format(err_mot_of))
                    #print(df_group)


                    # Setting the confidence as the iou score
                    # TODO
            else:
                print('All Tracks were initialized becaue there was no detection fot {} frames'.format(DET_GAP))

                #print(df_group)
            # Assign new tracks
            for t in df_group.index[df_group['track_id'] == -1].tolist():

                df_group.at[t, 'track_id'] = Track_id
                Track_id+=1


            #print(df_group)
            df_p_group = pd.DataFrame(columns=headers)
            df_p_group = df_p_group.append(df_group, ignore_index=True)
            df_p_group = df_p_group.dropna()
            #print(df_p_group)
            df_track = df_track.append(df_p_group, ignore_index=True)
            #print(df_track)
            if PLOT_FLAG:
                bbox =  bb.bbox_list_from_pandas(df_p_group)
                #print(df_p_group['track_id'].tolist())
                ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=[df_p_group['track_id'].tolist(),df_p_group['track_iou_score'].tolist()],ax=ax,title = "frame: "+str(f))

                if SAVE_FLAG:
                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
                    #print(np.shape(img))
                    cv.imwrite(os.path.join(results_dir,"{}.png".format(f)), img);
                    #cv.imshow('test',img)
                    if VID_FLAG:

                        out.write(img)
            #bbox_iou(bboxA, bboxB)

        #    END OF PROCESS
        if VID_FLAG:
            out.release()

        print('Number of Detections:')
        print(np.shape(df_track))

        if SAVE_FLAG:
            save_in = os.path.join(results_dir,"pred_tracks.pkl")
            df_track.to_pickle(save_in)
            csv_file = os.path.splitext(save_in)[0] + "1.csv"
            export_csv = df_track.to_csv(csv_file, sep='\t', encoding='utf-8')





if __name__ == '__main__':
    main()
