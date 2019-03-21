import matplotlib
matplotlib.use('TkAgg')

# For visulization

import matplotlib.pyplot as plt

import evaluation.evaluation_funcs as evf
import organize_code.code.utils as ut
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
import pandas as pd
import numpy as np

import os

# metric MOT - requires python3
import motmetrics as mm
COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
'#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def getIDF1(Pred,GT):
    """
    Input pandas
    Output metrics from MOTchallenge
    """
    # Loop on frames in PandaList
    # Assign Tp/Fp To the Panda PandaList
    acc = mm.MOTAccumulator(auto_id=True)
    Pred,GT = prepare_list(Pred,GT)
    # Loop on Frames:
    PPreds =Pred.groupby('frame')
    PGTs = GT.groupby('frame')
    print('# {} of bbox in Ground Truth'.format(len(GT)))
    print('# {} of bbox in Predictions'.format(len(Pred)))
    frames_ingt = PGTs.groups.keys()
    frames_inpred = PPreds.groups.keys()
    print(type(frames_inpred))
    all_frames = list(set(frames_ingt + frames_inpred)).sort()


    for f in all_frames:
        if f%50==0:
            print('frame: {}'.format(f))
        p = PPreds.get_group(f)
        g = PGTs.get_group(f)

        # track id :
        tk_p = p['track_id'].tolist()
        tk_g = g['track_id'].tolist()
        # Computing Distance
        pred_bbox = np.array(bb.bbox_list_MOT_from_pandas(p) ) # Format X, Y, Width, Height
        gt_bbox = np.array(bb.bbox_list_MOT_from_pandas(g) )

        dist = mm.distances.iou_matrix(gt_bbox,pred_bbox , max_iou=0.5)
        # Update list
        acc.update(
        tk_g,                 # Ground truth objects in this frame
        tk_p,                  # Detector hypotheses in this frame
        dist                # Distances from object 'a' to hypotheses 1, 2, 3
        )
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp','idf1'], name='acc')
    print(summary)
    return mh

def prepare_list(Pred,GT):
    Pred.loc[:,'metric'] = 'FP'
    Pred.loc[:,'GT_track_id'] = 0
    GT.loc[:,'metric'] = 'FN'
    GT.loc[:,'GT_track_id'] = 0
    Pred.loc[:,'matched'] = 0
    GT.loc[:,'matched'] = 0

    #if 'conf' not in list(Pred.head(0)):
    Pred.loc[:,'conf'] = 1.0

    if 'conf' not in list(GT.head(0)):
        GT.loc[:,'conf'] = 1.0


    headers_relevant = ['frame','conf','track_id',	'matched',	'metric','xmax','xmin',	'ymax',	'ymin','GT_track_id']
    GT =GT[headers_relevant]
    Pred =Pred[headers_relevant]
    return Perd,GT

def main():
    #import src
    OUTPUT_DIR ='../output'
    ROOT_DIR = '../'
    # Some constant for the script
    N = 1
    DET = 'TEST'
    EXP_NAME = '{}_N{}'.format(DET, N)
    TASK = 'task3'
    WEEK = 'week3'
    results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)


    #gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation.xml')
    gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation2.pkl')
    if os.path.isfile(gt_file):
        GT = pd.read_pickle(gt_file)
    # detection file can be txt/xml/pkl
    #det_file = os.path.join(results_dir,'kalman_out.pkl')
    det_file = os.path.join(results_dir,'pred_tracks.pkl')
    if os.path.isfile(det_file):
        Pred = pd.read_pickle(det_file)
    mh = getIDF1(Pred,GT)


######################

main()
