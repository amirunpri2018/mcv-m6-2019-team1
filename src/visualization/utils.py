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

# Visualization
import matplotlib

#from src.map import get_avg_precision_at_iou
import organize_code.utils_xml as utx

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
        l = range(len(l_bboxes))
        Ntext = 1

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

def bboxAnimation(movie_path,det_file,out_in = None,VID_FLAG = False):
    """
    Input:
    movie_path
        If movie path is a movie - load and extract frame - # TODO:
        If movie path is a folder - load images
    det_file : xml,txt or pkl
        panda list with the following column names
            ['frame','track_id', 'xmax', 'xmin', 'ymax', 'ymin']
    out_in
        folder to save output frames / movie -# TODO:

    """
    if os.path.isdir(movie_path):
        DIR_FLAG =True
    else:
        DIR_FLAG = False


    # Create folders if they don't exist
    if out_in is not None and not os.path.isdir(out_in):
        os.mkdir(out_in)

    # Get BBox detection from list
    df = ut.getBBox_from_gt(det_file)

    df.sort_values(by=['frame'])

    # Group bbox by frame
    df_grouped = df.groupby('frame')

    headers = list(df.head(0))
    print(headers )
    df_track = pd.DataFrame(columns=headers)

    first_frame = True

    for f, df_group in df_grouped:
        df_group = df_group.reset_index(drop=True)


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
            ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=df_group['track_id'].tolist(),ax=ax,title = str(f))

            if out_in is not None:
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

            #cv.imshow(img)
                if VID_FLAG:
                    fourcc = cv.cv.CV_FOURCC('M','J','P','G')

                    s = np.shape(img)
                    out = cv.VideoWriter(os.path.join(out_in,"IOU.avi"),fourcc, 30.0, (s[0],s[1]))
                    out.write(img)
            first_frame = False
            continue

        bbox =  bb.bbox_list_from_pandas(df_group)
        ut.plot_bboxes(cv.imread(im_path,cv.CV_LOAD_IMAGE_GRAYSCALE),bbox,l=[df_group['track_id'].tolist(),df_group['track_iou'].tolist()],ax=ax,title = "frame: "+str(f))

        if out_in is not None:
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(out_in,"{}.png".format(f)), img);

            if VID_FLAG:

                out.write(img)


    #    END OF PROCESS
    if VID_FLAG:
        out.release()
    return
