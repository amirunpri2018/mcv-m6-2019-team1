import matplotlib
matplotlib.use('TkAgg')

# For visulization

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import average_precision_score
import evaluation.evaluation_funcs as evf
import organize_code.code.utils as ut
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
import pandas as pd
import numpy as np
import random
import os

# metric MOT - requires python3
#import motmetrics as mm
COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
'#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def PandaTo_PR(PandaList):
    # get annotation - is the conf
    PLOT_FLAG = True
    print('Computing mAP for detection....')
    PandaList = PandaList.sort_values(['conf'], ascending=[0])
    conf = PandaList['conf'].tolist()
    #print(conf)

    met = PandaList['metric'].tolist()
    #print(len(set(conf)))
    TP = met.count('TP')
    FN = met.count('FN')
    FP = met.count('FP')
    print('# TP: {}'.format(TP))
    print('# FP: {}'.format(FP))
    print('# FN: {}'.format(FN))
    Ngt = met.count('TP')+met.count('FN')
    print(len(PandaList))
    print('# bbox in the GT: {}'.format(Ngt))
    mAP = list()
    # If all conf value is 1
    if len(set(conf)):
        print('All confedance are 1 the mean average preision is based on 10 random ranks')

        Niter = 10
        Shuffle_flag=True
    else:
        Niter = 1
        Shuffle_flag=False

    if PLOT_FLAG:
        fig = plt.figure(1)
        ax1 = plt.subplot(111)
        ax1.set(xlabel='Recall', ylabel='Precision',title='Prec-Recall Curve')
        ax1.grid()


    for it in range(Niter):
        if Shuffle_flag:
            print('Shuffle #{}'.format(it))
            random.shuffle(met)

        pre = list()
        recal = list()

        for d in range(len(met)):

            if met[d]=='TP':

                tp = float(met[0:d+1].count('TP'))
                fp = float(met[0:d+1].count('FP'))
                fn = float(met[0:d+1].count('FN'))

                pre.append(tp/(tp+fp))
                recal.append(tp/Ngt)

        # Sort according to recall
        map = pr2map(pre,recal,PLOT_FLAG=PLOT_FLAG,color=COLORS[it])
        print(map)
        # mean over the precision
        mAP.append(map)
    print(mAP)
    if PLOT_FLAG:
        plt.show()
    return np.mean(mAP)

def pr2map(pre,recal,PLOT_FLAG = False,color='b'):
    # Sort according to recall

    recal, pre = (list(t) for t in zip(*sorted(zip(recal, pre))))
    pre  = np.asarray(pre)
    # Align
    pre11 = []
    x = np.linspace(0.0, 1.0, 11)
    for recall_level in x:
        try:
            args = np.argwhere(recal >= recall_level).flatten()
            if args==[]:
                print('this is empty')
            prec = np.max(pre[args])
        except ValueError:
            prec = 0.0
        pre11.append(prec)
    print(pre11)
    # interpolate recall -11 steps
    #x=[0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1]
    #pre11 = np.interp(x, recal, pre)
    # mean over the precision
    if PLOT_FLAG:

        ax1 = plt.gca()
        ax1.plot(recal, pre,linestyle=':',color=color)
        ax1.scatter(x, pre11,marker='o',color=color)



    return np.mean(pre11)


def PandaTrack_PR(PandaList):
    # get annotation - is the conf
    PandaList = PandaList.sort_values(['conf'], ascending=[0])
    g_tk = PandaList.groupby('GT_track_id')
    mAP = list()
    id_full = PandaList['track_id'].tolist()
    for id,g_tk in g_tk:
        c_trk = g_tk['track_id'].value_counts().argmax()
        C = PandaList.query('track_id == {} or GT_track_id == {}'.format(c_trk,id))

        #print(C)
        #FP_list = PandaList.loc()
        conf = C['conf'].tolist()
        met = C['metric'].tolist()

        id_t = np.asarray(C['track_id'].tolist(),dtype=np.float64)
        id_tgt = np.asarray(C['GT_track_id'].tolist(),dtype=np.float64)
        Ngt = met.count('TP')+met.count('FN')
        #print('# bbox in the GT: {}'.format(Ngt))
        APi = list()
        # If all conf value is 1
        if len(set(conf)):
            print('All confedance are 1 the mean average preision is based on 10 random ranks')

            Niter = 10
            Shuffle_flag=True
        else:
            # iterate any way

            Niter = 10
            Shuffle_flag=True

        for it in range(Niter):
            if Shuffle_flag:
                print('Shuffle #{}'.format(it))
                random.shuffle(met)

            pre = list()
            recal = list()

            #g_tk.groupby(['track_id']).size()
            for d in range(len(met)):

                if id_t[d]==c_trk:

                    a = id_t[0:d+1]==c_trk
                    b = id_tgt[0:d+1]==id

                    tp = np.count_nonzero(a&b)

                    #print(tp)
                    #tp = float(id_t[0:d+1].count(c_trk))
                    # was detected as the same track - but it belong to another track id
                    fp = np.count_nonzero(a&(~b))
                    fn = np.count_nonzero((~a)&b)
                    #print(fp)
                    pre.append(tp/float(tp+fp))
                    recal.append(tp/float(Ngt))


            # Sort according to recall
            #print('pre:')
            #print(pre)
            #print('recall:')
            #print(recal)
            map = pr2map(pre,recal)
            print(map)
            # mean over the precision
            APi.append(map)
        mAP.append(np.mean(APi))
    #print(mAP)
    return np.mean(mAP),mAP

def PandaTpFp(Pred,GT ,iou_thresh = 0.5,save_in = None):
    # Loop on frames in PandaList
    # Assign Tp/Fp To the Panda PandaList

    Pred.loc[:,'metric'] = 'FP'
    Pred.loc[:,'GT_track_id'] = 0
    GT.loc[:,'metric'] = 'FN'
    GT.loc[:,'GT_track_id'] = 0
    #print(GT['conf'].tolist())
    #print(Pred['conf'].tolist())
    Pred.loc[:,'matched'] = 0
    GT.loc[:,'matched'] = 0

    #if 'conf' not in list(Pred.head(0)):
    Pred.loc[:,'conf'] = 1.0


    if 'conf' not in list(GT.head(0)):
        GT.loc[:,'conf'] = 1.0



    headers_relevant = ['frame','conf','track_id',	'matched',	'metric','xmax',	'xmin',	'ymax',	'ymin','GT_track_id']
    GT =GT[headers_relevant]
    Pred =Pred[headers_relevant]
    New_List = pd.DataFrame(columns=headers_relevant)

    PPreds =Pred.groupby('frame')
    PGTs = GT.groupby('frame')
    print('# {} of bbox in Ground Truth'.format(len(GT)))
    print('# {} of bbox in Predictions'.format(len(Pred)))
    frames_ingt = PGTs.groups.keys()
    frames_inpred = PPreds.groups.keys()
    na_in_pred = list(set(frames_ingt)-set(frames_inpred))
    print('No prediction in the following frame:')
    for na in na_in_pred:

        print(na)
        New_List.append(PGTs.get_group(na))
    for f,Pframe in PPreds:
        Pframe = Pframe.dropna()
        Pframe = Pframe.reset_index(drop=True)
        #idx = np.min(Pframe.index.values.tolist() )
        # Finding matching GT frame
        #df.loc[pd.IndexSlice[age,:], 'Raitings'].idxmin()[1]
        #print(GT)
        #if f>400:
        #    break
        if f%50==0:
            print('frame: {}'.format(f))
        if f not in frames_ingt:
            # assign False to all BBox in Predictions
            New_List = New_List.append(Pframe, ignore_index=True)
            continue

        Current_GT= PGTs.get_group(f)
        Current_GT = Current_GT.dropna()
        Current_GT = Current_GT.reset_index(drop=True)

        if Current_GT.empty:
            # assign False to all BBox in Predictions
            New_List = New_List.append(Pframe, ignore_index=True)
            continue


        iou_mat = bb.bbox_lists_iou(Pframe,Current_GT)
        matchlist,iou_score = bb.match_iou(iou_mat)

        for m,s in zip(matchlist,iou_score):



            if s>iou_thresh:
                Current_GT.loc[m[1],'matched'] = 1 # had a match
                Pframe.loc[m[0],'metric'] = 'TP'
                Pframe.loc[m[0],'GT_track_id'] = Current_GT.loc[m[1],'track_id']


        New_List = New_List.append(Pframe, ignore_index=True)
        Current_GTm =Current_GT.groupby('matched')

        if 0 in Current_GTm.groups.keys():
            #print(Current_GTm.get_group(0))
            New_List = New_List.append(Current_GTm.get_group(0), ignore_index=True)

        #print('#{} bbox in Evaluation List'.format(len(New_List)))
    print(np.shape(New_List))
    csv_file = os.path.splitext(save_in)[0] + "1.csv"
    export_csv = New_List.to_csv(csv_file, sep='\t', encoding='utf-8')
    New_List = New_List.dropna()
    print(np.shape(New_List))
    if save_in is not None:
        New_List.to_pickle(save_in)
        #csv_file = base = os.path.splitext(thisFile)[0]
        csv_file = os.path.splitext(save_in)[0] + ".csv"

        export_csv = New_List.to_csv(csv_file, sep='\t', encoding='utf-8')

    print(len(New_List))
    print(len(GT))
    return New_List

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
