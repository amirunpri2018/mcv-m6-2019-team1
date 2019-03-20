from sklearn.metrics import average_precision_score
import evaluation.evaluation_funcs as evf
import organize_code.code.utils as ut
#import organize_code.utils as ut
import evaluation.bbox_iou as bb
import pandas as pd
import numpy as np
import random



def PandaTo_PR(PandaList):
    # get annotation - is the conf
    print('Computing mAP for detection....')
    PandaList = PandaList.sort_values(['conf'], ascending=[0])
    conf = PandaList['conf'].tolist()
    #print(conf)

    met = PandaList['metric'].tolist()
    #print(len(set(conf)))
    Ngt = met.count('TP')+met.count('FN')
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
        map = pr2map(pre,recal)
        print(map)
        # mean over the precision
        mAP.append(map)
    print(mAP)
    return np.mean(mAP)

def pr2map(pre,recal):
    # Sort according to recall
    recal, pre = (list(t) for t in zip(*sorted(zip(recal, pre))))
    # interpolate recall -11 steps
    pre11 = np.interp([0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1], recal, pre)
    # mean over the precision

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
            Niter = 1
            Shuffle_flag=False

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
    GT.loc[:,'conf'] = 1.0
    #print(GT['conf'].tolist())
    #print(Pred['conf'].tolist())
    Pred.loc[:,'matched'] = 0
    GT.loc[:,'matched'] = 0
    headers = list(Pred.head(0))
    print(headers )
    New_List = pd.DataFrame(columns=headers)

    PPreds =Pred.groupby('frame')
    PGTs = GT.groupby('frame')

    for f,Pframe in PPreds:
        Pframe = Pframe.dropna()
        Pframe = Pframe.reset_index(drop=True)
        idx = np.min(Pframe.index.values.tolist() )
        # Finding matching GT frame
        #df.loc[pd.IndexSlice[age,:], 'Raitings'].idxmin()[1]
        #print(GT)
        #if f>400:
        #    break
        Current_GT= PGTs.get_group(f)
        Current_GT = Current_GT.dropna()
        Current_GT = Current_GT.reset_index(drop=True)

        idx_gt = np.min(Current_GT.index.values.tolist() )

        if Current_GT.empty:
            # assign False to all BBox in Predictions
            New_List = New_List.append(Pframe, ignore_index=True)
            continue

        else:
            iou_mat = bb.bbox_lists_iou(Current_GT,Pframe)
            matchlist,iou_score = bb.match_iou(iou_mat)
            for m,s in zip(matchlist,iou_score):

                Current_GT.loc[m[1],'matched'] = 1 # had a match
                #Pframe.loc[m[0],'matched'] = True
                if s>iou_thresh:
                    Pframe.loc[idx+m[0],'metric'] = 'TP'
                    Pframe.loc[idx+m[0],'GT_track_id'] = Current_GT.loc[idx_gt+m[1],'track_id']
                    #print(Current_GT)
                #else:
                    #Pframe.loc[m[0],'metric'] = False
            #print(Pframe)
            # For all the one with no match
            #idx_unmatched = Current_GT.index[Current_GT['matched']==False].tolist()
            # For all the FP - no bbox was found;
            New_List = New_List.append(Pframe, ignore_index=True)
            Current_GTm =Current_GT.groupby('matched')

            if np.any(Current_GTm.groups.keys()==0):
                #print(Current_GTm.get_group(0))
                New_List = New_List.append(Current_GTm.get_group(0), ignore_index=True)

    New_List = New_List.dropna()
    if save_in is not None:
        New_List.to_pickle(save_in)


    return New_List
