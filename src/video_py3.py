import matplotlib
matplotlib.use('TkAgg')

# For visulization

import matplotlib.pyplot as plt

#import organize_code.utils as ut
#import evaluation.bbox_iou as bb
import pandas as pd
pd.__version__
import numpy as np

import os

# metric MOT - requires python3
import motmetrics as mm
COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
'#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def convert_pkalman(df):
    #['img_id', 'boxes', 'track_id', 'scores']
    #-->['conf', 'frame', 'occlusion', 'track_id', 'xmax', 'xmin', 'ymax', 'ymin', 'track_iou', 'Dx', 'Dy', 'rot', 'zoom']

    df = df.rename(columns={"img_id": "frame"})
    if 'scores' in df.head(0):
         df = df.rename(columns={"scores": "conf"})
     else:
         df.loc[:,'conf']=1.0
    df.loc[:,'xmin']=0.0
    df.loc[:,'ymin']=0.0
    df.loc[:,'xmax']=0.0
    df.loc[:,'ymax']=0.0
    for i, row in df.iterrows():
        bbox =row['boxes']
        df.loc[i,'xmin'] = bbox[0]
        df.loc[i,'ymin'] = bbox[1]
        df.loc[i,'xmax'] = bbox[2]
        df.loc[i,'ymax'] = bbox[3]

    return df
def bbox_list_MOT_from_pandas(Pbbox):
    """
    Convert Pandas List to a simple List of BBOX
    """
    bbox = list()
    for bA in Pbbox.itertuples():#enumerate(lbboxA):

        bbox.append([getattr(bA, 'xmin'),getattr(bA, 'ymin'),getattr(bA, 'xmax')-getattr(bA, 'xmin'),getattr(bA, 'ymax')-getattr(bA, 'ymin')])

    return bbox

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
    frames_ingt = list(PGTs.groups.keys())
    frames_inpred = list(PPreds.groups.keys())

    all_frames = list(set(frames_ingt + frames_inpred))


    for f in all_frames:
        if f%50==0:
            print('frame: {}'.format(f))

        if f not in frames_ingt:
            g = []
            tk_g =[]
            gt_bbox = []
            print('No GT in frame #{} '.format(f))
            continue
        else:
            g = PGTs.get_group(f)
            # track id :
            tk_g = g['track_id'].tolist()
            gt_bbox = np.array(bbox_list_MOT_from_pandas(g) ) # Format X, Y, Width, Height


        if f not in frames_inpred:
            p = []
            tk_p = []
            pred_bbox = []
            print('No prediction in frame #{} '.format(f))
        else:
            p = PPreds.get_group(f)
            tk_p = p['track_id'].tolist()
            pred_bbox = np.array(bbox_list_MOT_from_pandas(p) )# Format X, Y, Width, Height



         # Computing Distance
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
    #Pred.loc[:,'conf'] = 1.0

    if 'conf' not in list(GT.head(0)):
        GT.loc[:,'conf'] = 1.0


    headers_relevant = ['frame','conf','track_id',	'matched',	'metric','xmax','xmin',	'ymax',	'ymin','GT_track_id']
    GT =GT[headers_relevant]
    Pred =Pred[headers_relevant]
    return Pred,GT

def main():
    print(pd.__version__)
    #import src
    OUTPUT_DIR ='../output'
    ROOT_DIR = '../'
    # Some constant for the script
    N = 1
    DET = 'GT_XML'
    EXP_NAME = '{}_N{}'.format(DET, N)
    TASK = 'task1'
    WEEK = 'week3'
    results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK, EXP_NAME)


    #gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation.xml')
    gt_file = os.path.join(ROOT_DIR,'data', 'm6-full_annotation2.pkl')

    if os.path.isfile(gt_file):
        GT = pd.read_pickle(gt_file)
    else:
        print('couldnt load :')
        print(gt_file)
    # detection file can be txt/xml/pkl
    #det_file = os.path.join(results_dir,'kalman_out.pkl')
    det_file = os.path.join(results_dir,'pred_tracks.pkl')
    if os.path.isfile(det_file):
        Pred = pd.read_pickle(det_file)

        if any(x == 'boxes' for x in list(Pred.head(0))) and not any(x == 'xmin' for x in list(Pred.head(0))):
            print('coverting pandas format..')
            Pred = convert_pkalman(Pred)
    else:
        print('couldnt load :')
        print(det_file)
    mh = getIDF1(Pred,GT)


######################

main()
