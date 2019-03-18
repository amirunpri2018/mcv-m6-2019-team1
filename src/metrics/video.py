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
    PandaList = PandaList.sort(['conf'], ascending=[0])
    conf = PandaList['conf'].tolist()
    print(conf)

    # 2 Classes Car / background : TP , FP , FN - Ground Truth with no detections
    met = PandaList['metric'].tolist()
    print(len(set(conf)))
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
        # interpolate recall -11 steps
        # mean over the precision
        #print(pre)
        #print(recal)
        Idx = np.argsort(recal)
        recal, pre = (list(t) for t in zip(*sorted(zip(recal, pre))))
        #print[Idx]
        #recal = recal[Idx]
        #pre = pre[Idx]

        pre11 = np.interp([0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1], recal, pre)
        mAP.append(np.mean(pre11))
    print(mAP)
    return np.mean(mAP)
    # Loop on frames in gt
    # List Format will be:
    # only TP/FN - no place for FP
    # TODO: List is the union of Prediction bbox and GT bbox ( to allow FP) so we have to classes Car and No_Car
    # [TP/F]

def PandaTpFp(Pred,GT ,iou_thresh = 0.5):
    # Loop on frames in PandaList
    # Assign Tp/Fp To the Panda PandaList

    Pred.loc[:,'metric'] = 'FP'
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
        idx = np.min(Pframe.index.values.tolist() )
        # Finding matching GT frame
        #df.loc[pd.IndexSlice[age,:], 'Raitings'].idxmin()[1]
        #print(GT)
        if f>400:
            break
        Current_GT=PGTs.get_group(f)
        #Current_GT = GT.iloc[GT.loc['frame' == f]]
        #print(Current_GT)
        #Current_GT = GT.iloc[GT.index([GT.frame == f])]
        #Current_GT = PGTs.loc[PGTs.IndexSlice[f,:]]
        Current_GT = Current_GT.dropna()
        Pframe = Pframe.dropna()
        if Current_GT.empty:
            #print(Current_GT)
            # assign False to all BBox in Predictions
            #df1.loc['a'] > 0
            New_List = New_List.append(Pframe, ignore_index=True)
            #Pred.loc[Pred.index[Pred['frame']==f].tolist(),'metric'] = False
            continue

        else:
            iou_mat = bb.bbox_lists_iou(Current_GT,Pframe)
            matchlist,iou_score = bb.match_iou(iou_mat)
            for m,s in zip(matchlist,iou_score):

                print(m)
                print(s)
                Current_GT.loc[m[1],'matched'] = 1 # had a match
                #Pframe.loc[m[0],'matched'] = True
                if s>iou_thresh:
                    Pframe.loc[idx+m[0],'metric'] = 'TP'
                #else:
                    #Pframe.loc[m[0],'metric'] = False
            print(Pframe)
            # For all the one with no match
            #idx_unmatched = Current_GT.index[Current_GT['matched']==False].tolist()
            # For all the FP - no bbox was found;
            New_List = New_List.append(Pframe, ignore_index=True)
            Current_GTm =Current_GT.groupby('matched')
            #print(Current_GTm.groups.keys())
            #print(np.any(Current_GTm.groups.keys()==0))
            if np.any(Current_GTm.groups.keys()==0):
                print(Current_GTm.get_group(0))
                New_List = New_List.append(Current_GTm.get_group(0), ignore_index=True)

    New_List = New_List.dropna()
    print(New_List)
    PandaTo_PR(New_List)
    return New_List
            #print(iou_score)
            #df_group.index[df_group['track_id'] == -1].tolist()
        #PGTs[]
        #bbox_list_from_pandas(Pframe)
    """
    Convert Pandas List to a simple List of BBOX
    """
    #bbox = list()
    #for bA in Pbbox.itertuples():#enumerate(lbboxA):

        #bbox.append([getattr(bA, 'ymin'),getattr(bA, 'xmin'),getattr(bA, 'ymax'),getattr(bA, 'xmax')])



        #bbox_lists_iou(lbboxA,lbboxB)
def mAP():
    map = evf.performance_accumulation_window(detections, annotations, iou_thresh=0.5)
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# average_precision_score(y_true, y_scores)
