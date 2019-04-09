import xml
import os
# -*- coding: utf-8 -*-

# Built-in modules
import logging
import urllib
import itertools
import cv2 as cv
# 3rd party modules
import bs4
import numpy as np
import pandas as pd

# Visualization
import matplotlib

#from src.map import get_avg_precision_at_iou
import organize_code.utils_xml as utx
import evaluation.bbox_iou as bb

matplotlib.use('TkAgg')


import matplotlib.patches as patches
# For visulization
import matplotlib.pyplot as plt
from skimage import exposure
#import src.evaluation.evaluation_funcs as evalf
# Local modules
import xml.etree.ElementTree as ET
# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def precision(tp, fp):
    """
    Computes precision.

    :param tp: True positives
    :param fp: False positives
    :return: Precision
    """
    return tp / (tp + fp)


def recall(tp, fn):
    """
    Computes recall.

    :param tp: True positives
    :param fn: False negatives
    :return: Recall
    """
    return tp / (tp + fn)


def fscore(tp, fp, fn):
    """
    Computes fscore.

    :param tp: True positives
    :param fp: False positives
    :param fn: False negatives
    :return: F-score
    """
    return 2 * precision(tp, fp) * recall(tp, fn) / (
                precision(tp, fp) + recall(tp, fn))


def destroy_bboxes(bboxes, prob=0.5):
    """
    Destroy bounding boxes based on probability value.

    :param bboxes: List of bounding boxes
    :param prob: Probability to dump a bounding box
    :return: List of bounding boxes
    """
    if not isinstance(bboxes, list):
        bboxes = list(bboxes)

    final_bboxes = list()
    for bbox in bboxes:
        if prob < np.random.random():
            final_bboxes.append(bbox)

    return final_bboxes


def create_bboxes(bboxes, shape, prob=0.5):
    """
    Create bounding boxes based on probability value.

    :param bboxes: List of bounding boxes
    :param prob: Probability to create a bounding box
    :return: List of bounding boxes
    """

    if not isinstance(bboxes, list):
        bboxes = list(bboxes)

    new_bboxes = bboxes[:]
    for bbox in bboxes:
        if prob > np.random.random():
            new_bbox = add_noise_to_bboxes(bbox, shape,
                                           noise_size=True,
                                           noise_size_factor=30.0,
                                           noise_position=True,
                                           noise_position_factor=30.0)
            new_bboxes.extend(new_bbox)

    return new_bboxes


def add_noise_to_bboxes(bboxes, shape, noise_size=True, noise_size_factor=5.0,
                        noise_position=True, noise_position_factor=5.0):
    """
    Add noise to a list of bounding boxes.

    Format of the bboxes is [[tly, tlx, bry, brx], ...], where tl and br
    indicate top-left and bottom-right corners of the bbox respectively.

    :param bboxes: List of bounding boxes
    :param noise_size: Flag to add noise to the bounding box size
    :param noise_size_factor: Factor noise in bounding box size
    :param noise_position: Flag to add noise to the bounding box position
    :param noise_position_factor: Factor noise in bounding box position
    :return: List of noisy bounding boxes
    """
    output_bboxes = np.array(bboxes).copy()
    output_bboxes = output_bboxes.tolist()
    # If there is only one bounding box, change to a list
    if not isinstance(output_bboxes[0], list):
        output_bboxes = [output_bboxes]

    # If noise in position, get a random pair of values. Else, (0, 0)
    if noise_position:
        change_position = np.random.randint(-noise_size_factor,
                                            high=noise_size_factor, size=2)
    else:
        change_position = (0, 0)

    # If noise in size, get a random pair of values. Else, (0, 0)
    if noise_size:
        change_size = np.random.randint(-noise_position_factor,
                                        high=noise_position_factor, size=2)
    else:
        change_size = (0, 0)

    # Iterate over the bounding boxes
    for bbox in output_bboxes:
        # If moving the bounding box in the vertical position does not exceed
        # the limits (0 and shape[0]), move the bounding box
        if not (bbox[0] + change_position[0] < 0 or
                shape[0] < bbox[0] + change_position[0] or
                bbox[2] + change_position[0] < 0 or
                shape[0] < bbox[2] + change_position[0]):
            bbox[0] = bbox[0] + change_position[0]
            bbox[2] = bbox[2] + change_position[0]
        # If moving the bounding box in the horizontal position does not exceed
        # the limits (0 and shape[1]), move the bounding box
        if not (bbox[1] + change_position[1] < 0 or
                shape[1] < bbox[1] + change_position[1] or
                bbox[3] + change_position[1] < 0 or
                shape[1] < bbox[3] + change_position[1]):
            bbox[1] = bbox[1] + change_position[1]
            bbox[3] = bbox[3] + change_position[1]

        # If reshaping the bounding box vertical axis does not exceed the
        # limits (0 and shape[0]), resize the bounding box
        for i in [0, 2]:
            if not (bbox[i] + change_size[0] < 0 or
                    shape[0] < bbox[i] + change_size[0]):
                bbox[i] = bbox[i] + change_size[0]
        # If reshaping the bounding box horizontal axis does not exceed the
        # limits (0 and shape[1]), resize the bounding box
        for i in [1, 3]:
            if not (bbox[i] + change_size[1] < 0 or
                    shape[1] < bbox[i] + change_size[1]):
                 bbox[i] = bbox[i] + change_size[1]

    # Return list of bounding boxes
    return output_bboxes


def bbox_iou(bbox_a, bbox_b):
    """
    Compute the intersection over union of two bboxes.

    Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    indicate top-left and bottom-right corners of the bbox respectively.

    determine the coordinates of the intersection rectangle.
    :param bbox_a: Bounding box
    :param bbox_b: Another bounding box
    :return: Intersection over union value
    """
    #ToDo: ESTO ES UN PARCHE HORRIBLE

    if bbox_a == bbox_b:
        iou = 1

    else:
        x_a = max(bbox_a[1], bbox_b[1])
        y_a = max(bbox_a[0], bbox_b[0])
        x_b = min(bbox_a[3], bbox_b[3])
        y_b = min(bbox_a[2], bbox_b[2])

        # Compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # Compute the area of both bboxes
        bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(bbox_a_area + bbox_b_area - inter_area)

    # Return the intersection over union value
    return iou


def iterate_iou(bboxes_result, bboxes_gt):

    iou = []
    for bbox, gt in itertools.product(bboxes_result, bboxes_gt):
        iou.append(bbox_iou(bbox, gt))

    iou = np.asarray(iou)

    return sum(iou) / len(iou)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.

    :param actual: A list of elements that are to be predicted (order doesn't
        matter)
    :param predicted: A list of predicted elements (order does matter)
    :param k: The maximum number of predicted elements. Default value: 10
    :return score: The average precision at k over the input lists
    """
    if k < len(predicted):
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.

    :param actual: A list of lists of elements that are to be predicted (order
       doesn't matter in the lists)
    :param predicted: A list of lists of predicted elements (order matters in
        the lists)
    :param k: The maximum number of predicted elements. Default value: 10
    :return score: The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def add_tag(parent, tag_name, tag_value=None, tag_attrs=None):
    """
    Add tag to a BeautifulSoup element.

    :param parent: Parent element
    :param tag_name: Tag name
    :param tag_value: Tag value. Default: None
    :param tag_attrs: Dictionary with tag attributes. Default: None
    :return: None
    """
    print("AAAAAAAA")
    print parent
    print tag_name
    tag = parent.new_tag(tag_name)
    if tag_value is not None:
        tag.string = tag_value
    if tag_attrs is not None and isinstance(tag_attrs, dict):
        tag.attrs = tag_attrs

    parent.annotations.append(tag)




def read_xml(xml_fname):
    """
    Read XML file.

    :param xml_fname: XML filename
    :return: BeautifulSoup object
    """
    xml_doc = urllib.urlopen(xml_fname)
    return bs4.BeautifulSoup(xml_doc.read(), features='lxml')


def get_bboxes_from_aicity(fnames):
    """
    Get bounding boxes from AICity XML-like file.

    :param fname: List XML filename
    :return: Pandas DataFrame with the data
    """
    if not isinstance(fnames, list):
        fnames = [fnames]

    bboxes = list()
    for fname in fnames:
        # Read file
        soup = read_xml(fname)
        # Get parent tag of bounding boxes
        bboxes_tag = soup.find('track')
        # Iterate over bounding boxes and append the attributes to the list
        for child in bboxes_tag.find_all('box'):
            bboxes.append(child.attrs)

    # Return DataFrame
    return pd.DataFrame(bboxes)

def get_bboxes_from_MOTChallenge(fname):
    """
    Get the Bboxes from the txt files
    MOTChallengr format [frame,ID,left,top,width,height,conf,-1,-1,-1]
     {'ymax': 84.0, 'frame': 90, 'track_id': 2, 'xmax': 672.0, 'xmin': 578.0, 'ymin': 43.0, 'occlusion': 1}
    fname: is the path to the txt file
    :returns: Pandas DataFrame with the data
    """
    f = open(fname,"r")
    BBox_list = list()

    for line in f:
        data = line.split(',')
        xmax = float(data[2])+float(data[4])
        ymax = float(data[3])+float(data[5])
        BBox_list.append({'frame':int(data[0]),'track_id':int(data[1]), 'xmin':float(data[2]), 'ymin':float(data[3]), 'xmax':xmax, 'ymax':ymax,'occlusion': 1,'conf' :float(data[6])})

    return pd.DataFrame(BBox_list)

def getBBox_from_gt(fname,save_in = None):
    if fname.endswith('.txt'):
        df = get_bboxes_from_MOTChallenge(fname)
    elif fname.endswith('.xml'):
        df = utx.get_bboxes_from_aicity_file(fname)
        df = df.rename(columns={"id": "track_id", "xbr": "xmax", "ybr": "ymax", "xtl": "xmin", "ytl": "ymin"}) #index=int,
        df[['track_id']] = df[['track_id']].astype(int)
        df[['frame']] = df[['frame']].astype(int)
        df.sort_values(by=['frame'])
        df = df.reset_index(drop=True)
    elif fname.endswith('.pkl'):
        df = pd.read_pickle(fname)
        #print(list(df.head(0)))

        if any(x == 'boxes' for x in list(df.head(0))) and not any(x == 'xmin' for x in list(df.head(0))):
            print('coverting pandas format..')
            df = convert_pkalman(df)
    print(fname + ' was loaded')

    if save_in is not None:
        df.to_pickle(save_in)
        #print(df.dtypes)
        #df[['frame', 'track_id']] = df[['frame', 'track_id']].astype(int)
        #df = df.astype({"frame": int})
        #print(df.dtypes)

    return df

def get_bboxes_from_pascal(fnames, track_id):
    """
    Get bounding boxes from Pascal XML-like file.

    :param fname: List XML filename
    :return: Pandas DataFrame with the data
    """
    if not isinstance(fnames, list):
        fnames = [fnames]

    bboxes = list()
    for fname in fnames:
        # Read file
        soup = read_xml(fname)
        #print(soup)
        # Get parent tag of bounding boxes
        object_tag = soup.find('object')
        attrs = {
            'frame': int(
                soup.find('filename').string.split('_')[1].split('.')[0]),
            'occlusion': int(object_tag.find('difficult').string),
            'track_id': track_id
        }

        # Iterate over bounding boxes and append the attributes to the list
        for child in object_tag.find('bndbox').children:
            if isinstance(child, bs4.element.Tag):
                attrs[child.name] = float(child.string)
        bboxes.append(attrs)

    # Return DataFrame
    return bboxes,pd.DataFrame(bboxes)


def get_files_from_dir(directory, excl_ext=None):
    """
    Get only files from directory.

    :param directory: Directory path
    :param excl_ext: List with extensions to exclude
    :return: List of files in directory
    """

    logger.debug(
        "Getting files in '{path}'".format(path=os.path.abspath(directory)))

    excl_ext = list() if excl_ext is None else excl_ext

    l = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.split('.')[
            -1] not in excl_ext
    ]

    logger.debug("Retrieving {num_files} files from '{path}'".format(
        num_files=len(l),
        path=os.path.abspath(directory)))

    return l

def mse(image_a, image_b):
    """
    Compute the Mean Squared Error between two images.

    The MSE is the sum of the squared difference between the two images.

    :param image_a:
    :param image_b:
    :return:
    """
    if image_a.shape != image_b.shape:
        raise ValueError('Images must have the same dimensions')

    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err


def non_max_suppression(bboxes, overlap_thresh):
    """
    Malisiewicz et al. method for non-maximum suppression.

    :param bboxes: List with bounding boxes
    :param overlap_thresh: Overlaping threshold
    :return: List of merger bounding boxes
    """

    bboxes = np.array(bboxes)

    # If there are no boxes, return an empty list
    if len(bboxes) == 0:
        return list()

    # If the bounding boxes integers, convert them to floats
    # This is important since we'll be doing a bunch of divisions
    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")

    # Initialize the list of picked indexes
    pick = list()

    # Grab the coordinates of the bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while 0 < len(idxs):
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap_thresh < overlap)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return bboxes[pick].astype("int")


def histogram(im_array, bins=128):
    """
    This function returns the centers of bins and does not rebin integer arrays. For integer arrays,
    each integer value has its own bin, which improves speed and intensity-resolution.

    This funcion support multi channel images, returning a list of histogram and bin centers for
    each channel.

    :param im_array: Numpy array representation of an image
    :param bins: Number of bins of the histogram
    :return: List of histograms and bin centers for each channel
    """

    hist, bin_centers = list(), list()
    for i in range(im_array.shape[2]):
        _hist, _bin_centers = exposure.histogram(im_array[..., i], bins)
        hist.append(_hist)
        bin_centers.append(_bin_centers)

    return hist, bin_centers


def compute_metrics(gt, img_shape, noise_size=True, noise_size_factor=5, noise_position=True,
                    noise_position_factor=5, create_bbox_proba=0.5, destroy_bbox_proba=0.5, k=10, iou_thresh=0.5):
    """
    1. Add noise to ground truth Bounding Boxes.
    2.Compute Fscore, IoU, Map of two lists of Bounding Boxes.

    :param gt: list of GT bounding boxes
    :param img_shape: Original image shape
    :param noise_size: Change bbox size param
    :param noise_position: Increase bbox size param
    :param destroy_bbox_proba: Proba of destroying Bboxes
    :param create_bbox_proba: Proba of creating Bboxes
    :param k: Map at k
    :return: Noisy Bboxes, Fscore, IoU, MaP
    """

    # Add noise to GT depending on noise parameter
    bboxes = add_noise_to_bboxes(gt, img_shape,
                                   noise_size=noise_size,
                                   noise_size_factor=noise_size_factor,
                                   noise_position=noise_position,
                                   noise_position_factor=noise_position_factor)

    # Randomly create and destroy bounding boxes depending
    # on probability parameter
    bboxes = create_bboxes(bboxes, img_shape, prob=create_bbox_proba)
    bboxes = destroy_bboxes(bboxes, prob=destroy_bbox_proba)

    bboxTP, bboxFN, bboxFP = evalf.performance_accumulation_window(bboxes, gt, iou_thresh)

    print(bboxTP, bboxFN, bboxFP )

    """
    Compute F-score of GT against modified bboxes PER FRAME NUMBER
    """
    # ToDo: Add dependency on frame number

    fscore_val = fscore(bboxTP, bboxFN, bboxFP)

    """
    Compute IoU of GT against modified Bboxes PER FRAME NUMBER:
    """

    iou = iterate_iou(bboxes, gt)

    """
    Compute mAP of GT against modified bboxes PER FRAME NUMBER:
    """
    map = mapk(bboxes, gt, k)

    return (bboxes, fscore_val, iou, map)

def convert_pkalman(df):
    #['img_id', 'boxes', 'track_id', 'scores']
    #-->['conf', 'frame', 'occlusion', 'track_id', 'xmax', 'xmin', 'ymax', 'ymin', 'track_iou', 'Dx', 'Dy', 'rot', 'zoom']

    df = df.rename(columns={"img_id": "frame"})
    if 'scores' in df.head():
        df = df.rename(columns={'scores':'conf'})
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


def get_files_from_dir2(cdir,ext = None):
    ListOfFiles = os.listdir(cdir)
    ListOfFiles.sort()
    file_list = []
    for file_name in ListOfFiles:
            #class_name = cdir.split('_')
        if file_name.endswith(ext):
            file_list.append(os.path.join(cdir,file_name))

    return file_list

def plot_traj(img, bboxes,traj=[],l=[],ax=None , title=''):
    if plt.gca() is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = plt.gca()
    #ax = plt.gca
    ax.cla()
    ax.imshow(img,cmap='gray')
    #print(l_bboxes)
    colors = 'bgrcmykw'

    # label as numbers
    i=0

    if l==[]:
        l = range(np.shape(bboxes)[0])
        #Ntext = 1

    if len(np.shape(l))==1:
        l=[l]

    Ntext = len(l)
    #tr = traj.groupby('track_id')
    #keys = tr.groups.keys

    for bbox in bboxes:

        tj  = traj[traj['track_id'] ==l[0][i] ]



        #bbox = int(bbox)
        bbox = np.asarray(bbox,dtype= np.int)

        cl_idx = int(l[0][i]) % 8
        color = colors[cl_idx]
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1, edgecolor=color, facecolor=color,alpha=0.2,hatch=r"//") #hatch=r"//",alpha=0.1,label=str(l[i]))
        rect2 = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1, edgecolor=color,fill=False)
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.set
        ax.add_patch(rect2)
        ax.set
        if len(tj)>0:
            ax.autoscale(False)
            ax.plot(tj['dy'].tolist()[0],tj['dx'].tolist()[0], linewidth=2, color=color,linestyle='-')
            ax.set
            ax.plot(tj['dyof'].tolist()[0],tj['dxof'].tolist()[0], linewidth=2, color=color,linestyle=':')
            ax.set
        plt.text(bbox[1],bbox[0],str(int(l[0][i])),{'color': 'w','fontweight':'bold'},bbox={'facecolor':color,'edgecolor':color, 'alpha':0.7,'pad':0})
        for t in range(1,Ntext):


            plt.text(bbox[1],bbox[2]+20,'{:0.2f}'.format(l[t][i]))


        i+=1
    ax.set_title(title)
    plt.pause(0.001)




        #ax.set
def plot_bboxes(img, bboxes,l=[],ax=None , title=''):
    #if ax is None:

    if plt.gca() is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = plt.gca()
    #ax = plt.gca
    ax.cla()
    ax.imshow(img,cmap='gray')
    #print(l_bboxes)
    colors = 'bgrcmykw'

    # label as numbers
    i=0

    if l==[]:
        l = range(np.shape(bboxes)[0])
        #Ntext = 1

    if len(np.shape(l))==1:
        l=[l]

    Ntext = len(l)
    for bbox in bboxes:

        #bbox = int(bbox)
        bbox = np.asarray(bbox,dtype= np.int)

        cl_idx = int(l[0][i]) % 8
        color = colors[cl_idx]
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1, edgecolor=color, facecolor=color,alpha=0.2,hatch=r"//") #hatch=r"//",alpha=0.1,label=str(l[i]))
        rect2 = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1, edgecolor=color,fill=False)
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.set
        ax.add_patch(rect2)
        ax.set

        plt.text(bbox[1],bbox[0],str(int(l[0][i])),{'color': 'w','fontweight':'bold'},bbox={'facecolor':color,'edgecolor':color, 'alpha':0.7,'pad':0})
        for t in range(1,Ntext):


            plt.text(bbox[1],bbox[2]+20,'{:0.2f}'.format(l[t][i]))


        i+=1
    ax.set_title(title)
    plt.pause(0.001)

def mean_velocity(tk,MOTION_MERGE):
    dx = tk['Dx'].iloc[-MOTION_MERGE:].mean()
    dy = tk['Dy'].iloc[-MOTION_MERGE:].mean()
    zoom = tk['zoom'].iloc[-MOTION_MERGE:].mean()
    rot = tk['rot'].iloc[-MOTION_MERGE:].mean()
    return [dx,dy,zoom,rot]

def track_cleanup(df,MIN_TRACK_LENGTH=5,STATIC_OBJECT = None,MOTION_MERGE = None):
    """
    This Function "clean up" detection according to a set of parameters
    df: panda list with the following columns names
    Mandatory:
    ----------
    ['frame','track_id', 'xmax', 'xmin', 'ymax', 'ymin']
    Optional:
    --------
    ['track_iou', 'Dx', 'Dy', 'rot', 'zoom']
    Not Relevant:
    ['conf']
    PARAMETERS:
    ==========

    1. MIN_TRACK_LENGTH =5
        remove tracks that are shorter than 5 frames

    2. STATIC_OBJECT - recomended 0.95
        remove tracks that has overlap of min 95% through all frames  -
        each bbox is compared to 1st bbox

    3. RATIO_VAR
        remove tracks with high variation of ratio (h/w)
        normalized by the length of the track
    4. MOTION_MERGE
        merge tracks that correspond with % of the expectence according to Dx,Dy,Zoom
    OUTPUT:
    df (Pandas list) without some of the detections
    """
    IMAGE_SIZE = (1080,1920)

    headers = list(df.head(0))
    #print(headers )
    #print('---------')
    if 'Mx' not in headers:
        # if the info is not exist -calculate it
        df_n = pd.DataFrame(columns=headers)
        tk_df = df.groupby('track_id')
        for id,tk in tk_df:
            tk = get_trackMotion(tk)
            df_n =df_n.append(tk,ignore_index=True)


        df = df_n
    if MOTION_MERGE is not None:
        print('merging tracks...')
        # connect tracks that has a continues motion



        # Calc expected trajectory (acc) based on N detections
        # add N synthetic bboxes to each track
        df.sort_values(by=['frame'])
        tk_df = df.groupby('track_id')
        syn = pd.DataFrame(columns=headers)
        for id,tk in tk_df:
            # acc - in Dx,Dy,Zoom,Rot in that order
            val = mean_velocity(tk,MOTION_MERGE)

            # get last frame
            nb = pd.DataFrame(columns=headers)
            #last_idx = tk.idxmax()
            #nb.loc[0]=tk.loc[tk['frame'].idxmax()]
            #print(tk.iloc[-1])
            nb.loc[0,'Mx'] = tk.iloc[-1]['Mx']
            nb.loc[0,'My'] = tk.iloc[-1]['My']
            nb.loc[0,'area'] = tk.iloc[-1]['area']
            nb.loc[0,'zoom'] = tk.iloc[-1]['zoom']
            nb.loc[0,'ratio'] = tk.iloc[-1]['ratio']
            nb.loc[0,'frame'] = tk.iloc[-1]['frame']
            nb.loc[0,'track_id']= tk.iloc[-1]['track_id']
            #nb = nb.reset_index()
            #last = tk.loc[tk['frame'].idxmax()]
            #last = last.reset_index()
            # add synthetic bbox
            for i in range(1,MOTION_MERGE):


                nb.loc[i,'Mx'] = nb.iloc[i-1]['Mx']+ val[0]
                nb.loc[i,'My'] = nb.iloc[i-1]['My']+ val[1]
                nb.loc[i,'area'] = nb.iloc[i-1]['area']* val[2]
                nb.loc[i,'ratio'] = nb.iloc[i-1]['ratio']*val[3]
                nb.loc[i,'frame'] = nb.iloc[i-1]['frame']+ 1
                nb.loc[i,'track_id']= nb.iloc[i-1]['track_id']
                # append the new DataFrame
            syn = syn.append(nb, ignore_index=True)
        syn = syn.reset_index()
        # check correspondence for each frame

        sy_fs = syn.groupby(['frame','track_id'])


        VAR = 0.70
        # Create a List with only 1st frame of each track
        df.sort_values('frame')
        df_tk = df.groupby('track_id')
        df1f = pd.DataFrame(list(df.head(0)))
        for t,d in df_tk:
            d = d.reset_index()
            df1f = df1f.append(d.loc[0])
        df_fs = df1f.groupby('frame')
        # Loop on synthetic bbox
        for f, sf in sy_fs:
            #print(f)
            mx_min = sf['Mx']-10
            mx_max = sf['Mx']+10
            my_min = sf['My']-10
            my_max = sf['My']+10
            size_min = sf['area']*VAR
            size_max = sf['area']*(2-VAR)
            ratio_min = sf['ratio']*VAR
            ratio_max = sf['ratio']*(2-VAR)


            if f[0] not in df_fs.groups.keys():
                continue

            c_df = df_fs.get_group(f[0])
            c_df = c_df.reset_index(drop=True)
            for i in range(len(c_df)):
                if f[1]==c_df.loc[i,'track_id']:
                    continue
                print(int(c_df.loc[i,'track_id']))
                if np.any(c_df.loc[i,'Mx'] >mx_min):
                    if np.any(c_df.loc[i,'Mx'] <mx_max):
                        if np.any(c_df.loc[i,'My'] >my_min):
                            if np.any(c_df.loc[i,'My'] <my_max):
                                if np.any(c_df.loc[i,'area'] >size_min):
                                    if np.any(c_df.loc[i,'area'] <size_max):
                                        if np.any(c_df.loc[i,'ratio'] >ratio_min):
                                            if np.any(c_df.loc[i,'ratio'] <ratio_max):
                                                print('Track {} and {} Track have been merge into {}'.format(f[1],int(c_df.loc[i,'track_id']),f[1]))
                                                df.loc[df['track_id'] == c_df.loc[i,'track_id'],'track_id'] = f[1]
                                                break

            # For each frame compare if the synthetic is similar to another track - if it is replace the entire track id
            # If all conditions met the first frame of the track - it is a match





    if MIN_TRACK_LENGTH is not None:

        print(np.shape(df)[0])
        mask = df.groupby('track_id').frame.transform('count') > MIN_TRACK_LENGTH
        df =df[mask]
        print('all Tracks shorter than {} frames -were remove'.format(MIN_TRACK_LENGTH))
        print(np.shape(df)[0])



    if STATIC_OBJECT is not None:

        df_g = df.groupby('track_id', as_index=False)
        trk_ids = list()
        for t, d in df_g:

            d = d.agg({'Mx':'std','My':'std'})

            if (d['Mx']+d['My'])/2 > STATIC_OBJECT:
                trk_ids.append(t)


        df =df[df['track_id'].isin(trk_ids)]


        print('all static Tracks with a std lower than {}-were remove'.format(STATIC_OBJECT))


    df_out = df.reset_index(drop=True)

    return df_out

def std2(x):
    print(x.head(0))
    print('printing std...')
    print(x)
    print('===================')
    x['Mx']=sum(x['Mx'])+sum(x['My'])/2
    return(y)

def OF_refine_trk(df,pic_dir,alpha=0.9):
    """
    This Function "OF_refine_trk" detection according to a set of parameters
    df: panda list with the following columns names
    Mandatory:
    ----------
    ['frame','track_id', 'xmax', 'xmin', 'ymax', 'ymin']
    Optional:
    --------
    ['track_iou', 'Dx', 'Dy', 'rot', 'zoom']
    Not Relevant:
    ['conf']
    INPUT:
    ==========

    1. MIN_TRACK_LENGTH =5
        remove tracks that are shorter than 5 frames

    2. frame_dir - folder contains the frames
        - in order to calculate the OF block matches


    3. RATIO_VAR
        remove tracks with high variation of ratio (h/w)
        normalized by the length of the track
    4. MOTION_MERGE
        merge tracks that correspond with % of the expectence according to Dx,Dy,Zoom
    OUTPUT:
    df (Pandas list) without some of the detections
    """
    IMAGE_SIZE = (1080,1920)

    headers = list(df.head(0))
    #print(headers )
    #print('---------')
    if MOTION_MERGE is not None:
        print('merging tracks...')
        # connect tracks that has a continues motion

        if 'Mx' not in headers:
            # if the info is not exist -calculate it
            df_n = pd.DataFrame(columns=headers)
            tk_df = df.groupby('track_id')
            for id,tk in tk_df:
                tk = get_trackMotion(tk)
                df_n =df_n.append(tk,ignore_index=True)


            df = df_n

        # Calc expected trajectory (acc) based on N detections
        # add N synthetic bboxes to each track
        df.sort_values(by=['frame'])
        tk_df = df.groupby('track_id')
        syn = pd.DataFrame(columns=headers)
        for id,tk in tk_df:
            # acc - in Dx,Dy,Zoom,Rot in that order
            val = mean_velocity(tk,MOTION_MERGE)

            # get last frame
            nb = pd.DataFrame(columns=headers)
            #last_idx = tk.idxmax()
            #nb.loc[0]=tk.loc[tk['frame'].idxmax()]
            #print(tk.iloc[-1])
            nb.loc[0,'Mx'] = tk.iloc[-1]['Mx']
            nb.loc[0,'My'] = tk.iloc[-1]['My']
            nb.loc[0,'area'] = tk.iloc[-1]['area']
            nb.loc[0,'zoom'] = tk.iloc[-1]['zoom']
            nb.loc[0,'ratio'] = tk.iloc[-1]['ratio']
            nb.loc[0,'frame'] = tk.iloc[-1]['frame']
            nb.loc[0,'track_id']= tk.iloc[-1]['track_id']
            #nb = nb.reset_index()
            #last = tk.loc[tk['frame'].idxmax()]
            #last = last.reset_index()
            # add synthetic bbox
            for i in range(1,MOTION_MERGE):


                nb.loc[i,'Mx'] = nb.iloc[i-1]['Mx']+ val[0]
                nb.loc[i,'My'] = nb.iloc[i-1]['My']+ val[1]
                nb.loc[i,'area'] = nb.iloc[i-1]['area']* val[2]
                nb.loc[i,'ratio'] = nb.iloc[i-1]['ratio']*val[3]
                nb.loc[i,'frame'] = nb.iloc[i-1]['frame']+ 1
                nb.loc[i,'track_id']= nb.iloc[i-1]['track_id']
                # append the new DataFrame
            syn = syn.append(nb, ignore_index=True)
        syn = syn.reset_index()
        # check correspondence for each frame

        sy_fs = syn.groupby(['frame','track_id'])


        VAR = 0.70
        # Create a List with only 1st frame of each track
        df.sort_values('frame')
        df_tk = df.groupby('track_id')
        df1f = pd.DataFrame(list(df.head(0)))
        for t,d in df_tk:
            d = d.reset_index()
            df1f = df1f.append(d.loc[0])
        df_fs = df1f.groupby('frame')
        # Loop on synthetic bbox
        for f, sf in sy_fs:
            #print(f)
            mx_min = sf['Mx']-10
            mx_max = sf['Mx']+10
            my_min = sf['My']-10
            my_max = sf['My']+10
            size_min = sf['area']*VAR
            size_max = sf['area']*(2-VAR)
            ratio_min = sf['ratio']*VAR
            ratio_max = sf['ratio']*(2-VAR)


            if f[0] not in df_fs.groups.keys():
                continue

            c_df = df_fs.get_group(f[0])
            c_df = c_df.reset_index(drop=True)
            for i in range(len(c_df)):
                if f[1]==c_df.loc[i,'track_id']:
                    continue
                print(int(c_df.loc[i,'track_id']))
                if np.any(c_df.loc[i,'Mx'] >mx_min):
                    if np.any(c_df.loc[i,'Mx'] <mx_max):
                        if np.any(c_df.loc[i,'My'] >my_min):
                            if np.any(c_df.loc[i,'My'] <my_max):
                                if np.any(c_df.loc[i,'area'] >size_min):
                                    if np.any(c_df.loc[i,'area'] <size_max):
                                        if np.any(c_df.loc[i,'ratio'] >ratio_min):
                                            if np.any(c_df.loc[i,'ratio'] <ratio_max):
                                                print('Track {} and {} Track have been merge into {}'.format(f[1],int(c_df.loc[i,'track_id']),f[1]))
                                                df.loc[df['track_id'] == c_df.loc[i,'track_id'],'track_id'] = f[1]
                                                break

            # For each frame compare if the synthetic is similar to another track - if it is replace the entire track id
            # If all conditions met the first frame of the track - it is a match

    df_out = df.reset_index(drop=True)

    return df_out


def get_trackMotion(dft,plot_flag = False):
    # Input df of 1 track
    dft.sort_values(by=['frame'])
    dft = dft.reset_index(drop=True)
    skip_first = True
    for i in dft.index.values.tolist():
        if skip_first:
            box_motion = bb.getMotionBbox(dft.ix[[i]],dft.ix[[i]])
            dft.loc[[i],'Mx'] = box_motion[4]
            dft.loc[[i],'My'] = box_motion[5]
            dft.loc[[i],'area'] = box_motion[6]
            dft.loc[[i],'ratio'] = box_motion[7]
            skip_first = False
            continue

        box_motion = bb.getMotionBbox(dft.ix[[i-1]],dft.ix[[i]])

        dft.loc[[i],'rot'] = box_motion[3]#.columns = ['Dy', 'Dx','zoom','rot','Mx','My']
        dft.loc[[i],'zoom'] = box_motion[2]
        dft.loc[[i],'Dx'] = box_motion[1]
        dft.loc[[i],'Dy'] = box_motion[0]
        dft.loc[[i],'Mx'] = box_motion[4]
        dft.loc[[i],'My'] = box_motion[5]
        dft.loc[[i],'area'] = box_motion[6]
        dft.loc[[i],'ratio'] = box_motion[7]

    df_out = dft
    if plot_flag:
        frames = dft['frame'].tolist()
        area = dft.loc[:,'area'].tolist()
        ratio = dft.loc[:,'ratio'].tolist()
        mx = dft.loc[:,'Mx'].tolist()
        my = dft.loc[:,'My'].tolist()
        fig = plt.figure()
        ax1= fig.add_subplot(211)
        ax2= fig.add_subplot(212)
        ax3 = ax2.twinx()
        ax4 = ax1.twinx()
        ax1.plot(frames,mx,':g')
        ax4.plot(frames,my,':b')
        ax2.plot(frames,area,':k')
        ax3.plot(frames,ratio,':m')
        ax2.set_ylabel('Area', color='k')
        ax3.set_ylabel('Ratio', color='m')
        ax1.set_ylabel('X', color='g')
        ax4.set_ylabel('Y', color='b')
        ax1.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='k')
        ax3.tick_params(axis='y', labelcolor='m')
        ax4.tick_params(axis='y', labelcolor='b')
        ax1.set_title('motion track #{}'.format(dft.loc[0,'track_id']))
        plt.show()

    return df_out
def bboxAnimation(movie_path,det_file,save_in = None,VID_FLAG = False,score_key ='conf'):
    """
    Input:
    movie_path
        If movie path is a movie - load and extract frame - # TODO:
        If movie path is a folder - load images
    det_file : xml,txt or pkl
        panda list with the following column names
            ['frame','track_id', 'xmax', 'xmin', 'ymax', 'ymin']
    save_in
        folder to save output frames / movie -# TODO:

    """
    if os.path.isdir(movie_path):
        DIR_FLAG =True
    else:
        DIR_FLAG = False


    # Create folders if they don't exist
    if save_in is not None and not os.path.isdir(save_in):
        os.mkdir(save_in)

    # Get BBox detection from list
    df = getBBox_from_gt(det_file)

    df.sort_values(by=['frame'])

    # Group bbox by frame
    df_grouped = df.groupby('frame')

    headers = list(df.head(0))
    print(headers )
    df_track = pd.DataFrame(columns=headers)

    first_frame = True
    #save_in_bkp = save_in
    #save_in = None
    for f, df_group in df_grouped:
        df_group = df_group.reset_index(drop=True)

        #if f>=300:
        #    save_in = save_in_bkp

        if DIR_FLAG:
            im_path = os.path.join(movie_path,'frame_'+str(int(f)).zfill(3)+'.jpg')
            if not os.path.isfile(im_path):
                break


        # 1st frame -
        if first_frame:

            plt.ion()
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111)

            bbox =  bb.bbox_list_from_pandas(df_group)
            plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=df_group['track_id'].tolist(),ax=ax,title = str(f))

            if save_in is not None:
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

            #cv.imshow(img)
                if VID_FLAG:
                    fourcc = cv.cv.CV_FOURCC('M','J','P','G')

                    s = np.shape(img)
                    out = cv.VideoWriter(os.path.join(save_in,"IOU.avi"),fourcc, 30.0, (s[0],s[1]))
                    out.write(img)
            first_frame = False
            continue

        bbox =  bb.bbox_list_from_pandas(df_group)
        headers = list(df.head(0))
        plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=[df_group['track_id'].tolist(),df_group[score_key].tolist()],ax=ax,title = "frame: "+str(f))

        if save_in is not None:
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(save_in,"{}.png".format(f)), img);

            if VID_FLAG:

                out.write(img)


    #    END OF PROCESS
    if VID_FLAG:
        out.release()
    return


def bboxAnimationOF(movie_path,det_file,save_in = None,VID_FLAG = False,score_key ='conf'):
    """
    Input:
    movie_path
        If movie path is a movie - load and extract frame - # TODO:
        If movie path is a folder - load images
    det_file : xml,txt or pkl
        panda list with the following column names
            ['frame','track_id', 'xmax', 'xmin', 'ymax', 'ymin']
    save_in
        folder to save output frames / movie -# TODO:

    """
    if os.path.isdir(movie_path):
        DIR_FLAG =True
    else:
        DIR_FLAG = False


    # Create folders if they don't exist
    if save_in is not None and not os.path.isdir(save_in):
        os.mkdir(save_in)

    # Get BBox detection from list
    df = getBBox_from_gt(det_file)

    df.sort_values(by=['frame'])

    # Group bbox by frame
    df_grouped = df.groupby('frame')

    # create trajectory of all track
    df_track = df.groupby('track_id')

    df_traj = pd.DataFrame({'track_id':[],'frame':[],'dx':[], 'dy':[], 'dxof':[], 'dyof':[]})
    for id,tt in df_track:
        print(id)

        df_t = pd.DataFrame({'track_id':[],'frame':[],'dx':[], 'dy':[], 'dxof':[], 'dyof':[]})
        f= tt['frame'].tolist()
        f = f[1:]
        dx_c = tt['Mx'].tolist()
        dx_c = dx_c[1:]
        dy_c = tt['My'].tolist()
        dyof_c = tt['ofDv'].tolist()
        dxof_c = tt['ofDu'].tolist()
        dyof_c = [i * -1 for i in dyof_c]
        dxof_c = [i * -1 for i in dxof_c]

        dy_c = dy_c[1:]
        dxof_c = dxof_c[1:]
        dyof_c = dyof_c[1:]
        dxof_c[0] = dx_c[0]
        dyof_c[0] = dy_c[0]

        dxof_c = np.cumsum(dxof_c)
        dyof_c = np.cumsum(dyof_c)

        for ii in range(1,len(dx_c)) :
            c_df = pd.DataFrame({'track_id':[],'frame':[],'dx':[], 'dy':[], 'dxof':[], 'dyof':[]})
            c_df['frame'] = [f[ii]]
            c_df['track_id']= [id]
            c_df['dx']= [dx_c[:ii]]
            c_df['dy']= [dy_c[:ii]]
            c_df['dxof']= [dxof_c[:ii]]
            c_df['dyof']= [dyof_c[:ii]]


            df_t =df_t.append(c_df)
        df_traj = df_traj.append(df_t)
    #print(df_traj)
    df_traj_tr = df_traj.groupby('frame')
    print(df_traj['frame'])
    headers = list(df.head(0))
    print(headers )
    df_track = pd.DataFrame(columns=headers)

    first_frame = True
    #save_in_bkp = save_in
    #save_in = None
    for f, df_group in df_grouped:
        df_group = df_group.reset_index(drop=True)

        #if f>=300:
        #    save_in = save_in_bkp

        if DIR_FLAG:
            im_path = os.path.join(movie_path,'frame_'+str(int(f)).zfill(3)+'.jpg')
            if not os.path.isfile(im_path):
                break


        # 1st frame -
        if first_frame:

            plt.ion()
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111)

            bbox =  bb.bbox_list_from_pandas(df_group)
            plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=df_group['track_id'].tolist(),ax=ax,title = str(f))

            if save_in is not None:
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

            #cv.imshow(img)
                if VID_FLAG:
                    fourcc = cv.cv.CV_FOURCC('M','J','P','G')

                    s = np.shape(img)
                    out = cv.VideoWriter(os.path.join(save_in,"IOU.avi"),fourcc, 30.0, (s[0],s[1]))
                    out.write(img)
            first_frame = False
            continue

        bbox =  bb.bbox_list_from_pandas(df_group)
        headers = list(df.head(0))


        if f in df_traj_tr.groups.keys():

            tj = df_traj_tr.get_group(f)
            plot_traj(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE), bbox,traj=tj,l=[df_group['track_id'].tolist(),df_group[score_key].tolist()],ax=ax , title= "frame: "+str(f))
        else:
            plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=[df_group['track_id'].tolist(),df_group[score_key].tolist()],ax=ax,title = "frame: "+str(f))
        if save_in is not None:
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(save_in,"{}.png".format(f)), img);

            if VID_FLAG:

                out.write(img)


    #    END OF PROCESS
    if VID_FLAG:
        out.release()
    return

def positive_integer(number):
    """
    Convert a number to an positive integer if possible.
    :param number: The number to be converted to a positive integer.
    :return: The positive integer
    :raise: argparse.ArgumentTypeError if not able to do the conversion
    """
    try:
        integ = int(number)
        if integ >= 0:
            return integ
        else:
            raise argparse.ArgumentTypeError('%s is not a positive integer' % number)
    except ValueError:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % number)

def show_gray_img(asd):
    plt.imshow(asd, cmap='gray')
    plt.show()


def show_quiver(x_component_arrows, y_components_arrows):
    plt.quiver(x_component_arrows, y_components_arrows)
    plt.show()


def subarray(array, (upper_left_pix_row, upper_left_pix_col), (lower_right_pix_row, lower_right_pix_col)):
    """
    Return a subarray containing the pixels delimited by the pixels between upper_left_pix and lower_right.
    If asked for pixels outside the image boundary, such pixels have value 0.
    """

    if upper_left_pix_row > lower_right_pix_row or upper_left_pix_col > lower_right_pix_col:
        raise ValueError('coordinates of the subarray should correspond to a meaningful rectangle')

    orig_array = np.array(array)

    num_rows = lower_right_pix_row - upper_left_pix_row + 1
    num_cols = lower_right_pix_col - upper_left_pix_col + 1

    subarr = np.zeros((num_rows, num_cols), dtype=orig_array.dtype)

    # zoomed outside the original image
    if lower_right_pix_col < 0 or lower_right_pix_row < 0 or \
                    upper_left_pix_col > orig_array.shape[1] - 1 or upper_left_pix_row > orig_array.shape[0] - 1:
        return subarr

    # region of the original image that is inside the desired region
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_o_1, i_o_1)              |    |
    # |   |             (j_o_2, i_o_2) |    |
    # |___|____________________________|    |
    # |                                 |
    # |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_o_1 = 0
    else:
        i_o_1 = upper_left_pix_col
    if upper_left_pix_row < 0:
        j_o_1 = 0
    else:
        j_o_1 = upper_left_pix_row

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_o_2 = orig_array.shape[1] - 1
    else:
        i_o_2 = lower_right_pix_col
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_o_2 = orig_array.shape[0] - 1
    else:
        j_o_2 = lower_right_pix_row


    # region of the final image that is inside the original image, and whose content will be taken from the orig im
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_f_1, i_f_1)              |    |
    # |   |             (j_f_2, i_f_2) |    |
    # |___|____________________________|    |
    #     |                                 |
    #     |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_f_1 = -upper_left_pix_col
    else:
        i_f_1 = 0
    if upper_left_pix_row < 0:
        j_f_1 = -upper_left_pix_row
    else:
        j_f_1 = 0

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_f_2 = (orig_array.shape[1] - 1) - upper_left_pix_col
    else:
        i_f_2 = num_cols - 1
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_f_2 = (orig_array.shape[0] - 1) - upper_left_pix_row
    else:
        j_f_2 = num_rows - 1

    subarr[j_f_1:j_f_2 + 1, i_f_1:i_f_2 + 1] = orig_array[j_o_1:j_o_2 + 1, i_o_1:i_o_2 + 1]

    return subarr

def obtain_timeoff_fps(ROOT_DIR,sequence, camera):

    """Input: Sequence number, Camera number
    Output: Time offset, fps"""

    tstamp_path = os.path.join(ROOT_DIR,'info', 'cam_timestamp', sequence + '.txt')

    with open(tstamp_path) as f:
        lines = f.readlines()
        for l in lines:
            cam, time_off = l.split()
            if cam == 'c015':
                fps = 8.0
            else:
                fps = 10.0
            if camera == cam:
                return np.float(time_off), np.float(fps)

def timestamp_calc(frame, time_offset, fps):
    return frame / fps + time_offset
