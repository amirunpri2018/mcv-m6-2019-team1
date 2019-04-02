import xml
import os
# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os
import urllib
import itertools

# 3rd party modules
import bs4
import numpy as np
import pandas as pd
#image
import cv2 as cv
#import imageio
from skimage import exposure

import src.evaluation.evaluation_funcs as evalf
# Local modules

# sort
import re
# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def readOF(OFdir, filename):
    """
    Reading Optical flow files
    0 Dim validation
    1 Dim u
    2 Dim v
    """
    # Sequance 1
    OF_path = os.path.join(OFdir, filename)
    OF = cv.imread(gt1_path, -1)
    u = (OF[:, :, 1].ravel() - 2 ** 15) / 64.0
    v = (OF[:, :, 2].ravel() - 2 ** 15) / 64.0
    valid_OF = OF[:, :, 0].ravel()
    u = np.multiply(u, valid_OF)
    v = np.multiply(v, valid_OF)
    return u, v

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


def get_files_from_dir2(cdir,ext = None):
    ListOfFiles = os.listdir(cdir)
    ListOfFiles.sort()
    file_list = []
    for file_name in ListOfFiles:
            #class_name = cdir.split('_')
        if file_name.endswith(ext):
            file_list.append(os.path.join(cdir,file_name))

    # sorting with respect to frame number
    file_list.sort(key=natural_keys)
    return file_list

def getBBox_from_gt(fname):
    if fname.endswith('.txt'):
        df = get_bboxes_from_MOTChallenge(fname)
    elif fname.endswith('.xml'):
        df = get_bboxes_from_MOTChallenge(fname)

def get_bboxes_from_MOTChallenge(fname):
    """
    Get the Bboxes from the txt files
    MOTChallengr format [frame,ID,left,top,width,height,1,-1,-1,-1]
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
        #BBox_list.append({'frame':int(data[0]),'track_id':int(data[1]), 'xmin':float(data[2]), 'ymin':float(data[3]), 'xmax':xmax, 'ymax':ymax,'occlusion': 1,'conf' :float(data[6])})
        BBox_list.append({'frame':int(data[0]),'track_id':int(data[1]), 'xmin':float(data[2]), 'ymin':float(data[3]), 'xmax':xmax, 'ymax':ymax,'occlusion': 1,'conf' :float(data[6])})
    return pd.DataFrame(BBox_list)

def frameIdfrom_filename(file_name):
    #file_name.split()
    return int(file_name.split('_')[-1].split('.')[0])


def getbboxmask(BboxList,frm,imsize):
    """
    Returns a mask of
    - False in the Background
    - True in the Foreground

    """

    bs = BboxList.loc[BboxList['frame']==frm]
    mask = np.zeros(imsize,dtype =bool)
    bbox_list = list()
    if not bs.empty:
        for b in bs.itertuples():
            #print(b)
            xmin = int(getattr(b, "xmin"))
            xmax = int(getattr(b, "xmax"))
            ymin = int(getattr(b, "ymin"))
            ymax = int(getattr(b, "ymax"))
            xx, yy = np.meshgrid( range(xmin,xmax), range(ymin,ymax) )

            bbox_list.append([xmin,ymin,xmax,ymax])
            # bbox_list.append([xmin,xmax,ymin,ymax])
            mask[yy,xx] = np.ones(np.shape(xx),dtype =bool)

    return mask,bbox_list

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def getImg_D(im_path,D=1,color_space=None,color_channels=[]):
    # color_channels len ==D


    if D==1 and color_space is None:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    I = cv.imread(im_path,Clr_flag)

    if color_space:
        I = cv.cvtColor(I,color_space)

    if not color_channels==[]:
        reduceI = I[...,color_channels]
        I = reduceI

    if D==1 and color_space is None:
        I = np.repeat(I[:, :, np.newaxis], D, axis=2)


    return I

def plot_bboxes(img, l_bboxes,l=[],ax=None , title=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    colors = 'bgrcmykw'

    # label as numbers
    i=0
    if l==[]:
        l = range(len(l_bboxes))

    for bboxes in l_bboxes:
        color = colors[np.random.choice(len(colors))]
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor=color,alpha=0.1,label=str(l[i]))
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.set
            plt.text(bbox[0],bbox[1],str(l[i]),{'color': color})

            i+=1
