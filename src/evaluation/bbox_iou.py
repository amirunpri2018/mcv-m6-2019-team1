from __future__ import division

import pandas as pd
import numpy as np

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    #ToDo: ESTO ES UN PARCHE HORRIBLE PERO IOU DABA NAN!
    #print('a', bboxA)
    #print('b',bboxB)
    #break
    if bboxA == bboxB:
        iou = 1

    else:
        # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
        # indicate top-left and bottom-right corners of the bbox respectively.

        # determine the coordinates of the intersection rectangle
        xA = max(bboxA[1], bboxB[1])
        yA = max(bboxA[0], bboxB[0])
        xB = min(bboxA[3], bboxB[3])
        yB = min(bboxA[2], bboxB[2])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both bboxes
        bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
        bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

        iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou

def bbox_list_from_pandas(Pbbox):
    """
    Convert Pandas List to a simple List of BBOX
    """
    bbox = list()
    for bA in Pbbox.itertuples():#enumerate(lbboxA):

        bbox.append([getattr(bA, 'ymin'),getattr(bA, 'xmin'),getattr(bA, 'ymax'),getattr(bA, 'xmax')])

    return bbox

def bbox_list_MOT_from_pandas(Pbbox):
    """
    Convert Pandas List to a simple List of BBOX
    """
    bbox = list()
    for bA in Pbbox.itertuples():#enumerate(lbboxA):

        bbox.append([getattr(bA, 'xmin'),getattr(bA, 'ymin'),getattr(bA, 'xmax')-getattr(bA, 'xmin'),getattr(bA, 'ymax')-getattr(bA, 'ymin')])

    return bbox
def bbox_lists_iou(lbboxA,lbboxB):
    """
    # This function receive the lists in a pandas format
    Return
    - iou: a np matrix size (len(A),len(B)) with all the iou scores


    """
    #print(lbboxA)

    iou = np.zeros((len(lbboxA),len(lbboxB)))
    i = 0
    for bA in lbboxA.itertuples():#enumerate(lbboxA):
        #print(bA)
        #bboxA = bA[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
        bboxA = [getattr(bA, 'ymin'),getattr(bA, 'xmin'),getattr(bA, 'ymax'),getattr(bA, 'xmax')]

        j=0
        for bB in lbboxB.itertuples():#enumerate(llbboxB):
            #print(bB)
            bboxB = [getattr(bB, 'ymin'),getattr(bB, 'xmin'),getattr(bB, 'ymax'),getattr(bB, 'xmax')]
            #bboxB = bB[['ymin', 'xmin', 'ymax', 'xmax']].values.tolist()

            iou[i][j] = bbox_iou(bboxA, bboxB)
            #iou[i][j] = bbox_iou(bboxA, bboxB)
            j+=1

        i+=1

    return iou


def match_iou(iou_mat,iou_th=0):
    """
    find the best match for the 1st dimension
    The order is important
    Returns:

    if 1st DIM > 2nd DIM
    The object with the lower score will be declared as terminated
    ---> in the Future - declared as occluded - if after N frames it is still occluded - the track will be over

    """
    #np.max(iou_mat)
    iou_mat_c = iou_mat.copy()
    match = list()
    iou_score = list()
    s = iou_mat.shape
    for i in range(np.min(s)):


        if np.count_nonzero(iou_mat)==0:
        #if iou_mat.size ==0:
            break

        ind = np.unravel_index(np.argmax(iou_mat_c, axis=None), s)
        iou_score.append(iou_mat_c[ind])
        if iou_score[i]>iou_th:

            match.append([ind[0],ind[1]])
        # remove i and j from iou_mat

        for r in range(s[0]):
            iou_mat_c[(r,ind[1])] = 0
        for c in range(s[1]):
            iou_mat_c[(ind[0],c)] = 0


    return match,iou_score
def calcCenterBbox(bbox):
    """
    bboxB = ['ymin', 'xmin','ymax','xmax']
    Returns y-center, x- center
    """
    return [bbox[0]+(bbox[2]-bbox[0])/2.0, bbox[1]+(bbox[3]-bbox[1])/2.0]

def calcAreaBbox(bbox):
    return float((bbox[2]-bbox[0])* (bbox[3]-bbox[1]))

def calcRatioBbox(bbox):
    return float(bbox[2]-bbox[0])/ float(bbox[3]-bbox[1])

def getMotionBbox(bAp,bBp):
    """
    Calc Motion simple features from sequantial bbox
    B with respect to A
    Returns:
    Dy,Dx,Zoom,Rot
    """

    bA = bbox_list_from_pandas(bAp)
    bB = bbox_list_from_pandas(bBp)

    bAc = calcCenterBbox(bA[0])
    bBc = calcCenterBbox(bB[0])

    D = list(np.array(bAc) - np.array(bBc))

    #D = bBc-bAc
    bAa = calcAreaBbox(bA[0])
    bBa = calcAreaBbox(bB[0])
    Ar = bBa/bAa

    bAr = calcRatioBbox(bA[0])
    bBr = calcRatioBbox(bB[0])
    Rr = bBr/bAr

    bBp['Dy'] = D[0]
    bBp['Dx'] = D[1]
    bBp['zoom'] = Ar
    bBp['rot'] = Rr
    return [D[0],D[1],Ar,Rr,bBc[0],bBc[1],bBa,bBr]
