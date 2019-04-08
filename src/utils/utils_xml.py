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

#import imageio
from skimage import exposure
#import src.evaluation.evaluation_funcs as evalf




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
    return pd.DataFrame(bboxes)

def read_xml(xml_fname):
    """
    Read XML file.

    :param xml_fname: XML filename
    :return: BeautifulSoup object
    """
    xml_doc = urllib.urlopen(xml_fname)
    return bs4.BeautifulSoup(xml_doc.read(), features='xml')


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
        bboxes_tag = soup.find_all('track')
        for trk in soup.find_all('track'):
            print(trk.get('id'))
            id = trk.get('id')
            #bboxes.append('track_id'
            # Iterate over bounding boxes and append the attributes to the list
            for child in trk.find_all('box'):
                dict_list=child.attrs
                dict_list['track_id'] = id
                bboxes.append(dict_list)


    # Return DataFrame
    return pd.DataFrame(bboxes)

def get_bboxes_from_aicity_file(fname, save_in=None):
    """
    Get bounding boxes from AICity XML-like file.

    :param fname: XML filename
    :param save_in: Filepath to save the DataFrame as a pickle file
    :return: Pandas DataFrame with the data
    """

    # Read file
    soup = read_xml(fname)

    # Create the main DataFrame
    df = pd.DataFrame()

    # Iterate over track tags
    for track_tag in soup.find_all('track'):
        # Get track tag attributes
        track_attrs = track_tag.attrs
        # List to store the corresponding bounding boxes
        bboxes = list()
        for bbox in track_tag.find_all('box'):
            bbox.attrs = dict((k, float(v)) for k, v in bbox.attrs.iteritems())
            bboxes.append(bbox.attrs)

        # Convert the results to a DataFrame and add track tag attributes
        df_track = pd.DataFrame(bboxes)
        for k, v in track_attrs.iteritems():
            df_track[k] = v

        # Append data to the main DataFrame
        df = df.append(df_track)
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Save DataFrame if necessary
    if save_in is not None:
        df.to_pickle(save_in)

    # Return DataFrame
    return df
def add_tag(parent, tag_name, tag_value=None, tag_attrs=None):
    """
    Add tag to a BeautifulSoup element.

    :param parent: Parent element
    :param tag_name: Tag name
    :param tag_value: Tag value. Default: None
    :param tag_attrs: Dictionary with tag attributes. Default: None
    :return: None
    """
    tag = parent.new_tag(tag_name)
    if tag_value is not None:
        tag.string = tag_value
    if tag_attrs is not None and isinstance(tag_attrs, dict):
        tag.attrs = tag_attrs
    parent.annotations.append(tag)


def create_aicity_xml_file(fname, dataframe):
    """
    I checked and it doesnt work
    """
    soup = bs4.BeautifulSoup('<annotations>', 'xml')
    add_tag(soup, 'version', tag_value='1.1')
    add_tag(soup, 'meta')
    add_tag(soup.meta, 'task')
    add_tag(soup.meta.task, 'id', tag_value=2)
    add_tag(soup.meta.task, 'name', tag_value='AI_CITY_S03_C010')
    add_tag(soup.meta.task, 'size', tag_value=2141)
    add_tag(soup.meta.task, 'mode', tag_value='interpolation')
    add_tag(soup.meta.task, 'overlap', tag_value=5)
    add_tag(soup.meta.task, 'bugtracker')
    add_tag(soup.meta.task, 'flipped', tag_value=False)
    add_tag(soup.meta.task, 'created',
            tag_value='2019-02-26 14:46:50.754264+03:00')
    add_tag(soup.meta.task, 'updated',
            tag_value='2019-02-26 15:58:28.473275+03:00')
    add_tag(soup.meta.task, 'source', tag_value='vdo.avi')
    # TODO
    add_tag(soup.meta.task, 'labels')
    add_tag(soup.meta.task.labels, 'label')
    # TODO
    add_tag(soup.meta.task, 'segments')
    add_tag(soup.meta.task.segments, 'segment')

    add_tag(soup.meta.task, 'owner')
    add_tag(soup.meta.task.owner, 'username', tag_value='admin')
    add_tag(soup.meta.task.owner, 'email', tag_value='jrhupc@gmail.com')

    add_tag(soup.meta.task, 'original_size')
    add_tag(soup.meta.task.original_size, 'width', tag_value=1920)
    add_tag(soup.meta.task.original_size, 'height', tag_value=1080)

    add_tag(soup.meta, 'dumped', tag_value='2019-02-26 15:58:30.413447+03:00')

    add_tag(soup, 'track', tag_attrs={'id': 1, 'label': 'bycicle'})

    for bbox in dataframe.iteritems():
        add_tag(soup.track, 'bbox', tag_attrs={
            'frame': bbox.get('frame', None),
            'xtl': bbox.get('xtl', None),
            'ytl': bbox.get('ytl', None),
            'xbr': bbox.get('xbr', None),
            'ybr': bbox.get('ybr', None),
            'outside': bbox.get('outside', None),
            'occluded': bbox.get('occluded', None),
            'keyframe': bbox.get('keyframe', None),
        })

    output = soup.prettify()
    with (fname, 'wb') as f:
        f.writelines(output)

def add_track2xml(xml_fname,xml_new_fname,track,class_name):
    soup = read_xml(xml_fname)

    tag_meta=soup.find("annotations")
    track_idx=len(tag_meta.findChildren("track",recursive=False))+1
    order_idx=len(tag_meta.findChildren())

    new_track = create_tag(soup, "track", tag_attrs={'id': track_idx, 'label': class_name})

    i=0

    track = track.sort('frame')

    for bbox in track.itertuples():

        #frame  occlusion  track_id  xmax  xmin  ymax  ymin

        bbox_new = create_tag(soup,'box', tag_attrs= {
            'frame': getattr(bbox, "frame"),
            'xtl': getattr(bbox, "xmin"),
            'ytl': getattr(bbox, "ymin"),
            'xbr': getattr(bbox, "xmax"),
            'ybr': getattr(bbox, "ymax"),
            'outside': 0,
            'occluded': getattr(bbox, "occlusion"),
            'keyframe': 1,
        })
        new_track.insert(i,bbox_new)

        i+=1
    tag_meta.insert(order_idx,new_track)
    soup = soup.prettify()
    f = open(xml_new_fname, "w")
    f.write(soup)
    f.close()

def folderPascal2xml(xml_fname,xml_fname2,pdir):
    file_list = []
    i = 0
    ListOfDir = os.listdir(pdir)
    ListOfDir.sort()

    for cdir in ListOfDir:
        class_name = cdir.split('_')
        cdir = os.path.join(pdir,cdir)

        #print(int(class_name[0]))
        if os.path.isdir(cdir):

            file_list = []

            for file_name in os.listdir(cdir):
                if file_name.endswith(".xml"):
                    file_list.append(os.path.join(cdir,file_name))

            #print file_list

            bboxes,pd_bboxes = get_bboxes_from_pascal(file_list, int(class_name[0]))

            add_track2xml(xml_fname,xml_fname2,pd_bboxes,class_name[1])
            xml_fname = xml_fname2
            print(cdir + ' was added....')




def create_tag(parent, tag_name, tag_value=None, tag_attrs=None):
    """
    Add tag to a BeautifulSoup element.

    :param parent: Parent element
    :param tag_name: Tag name
    :param tag_value: Tag value. Default: None
    :param tag_attrs: Dictionary with tag attributes. Default: None
    :return: None
    """

    tag = parent.new_tag(tag_name)

    if tag_value is not None:
        tag.string = tag_value

    if tag_attrs is not None and isinstance(tag_attrs, dict):
        tag.attrs = tag_attrs

    return tag
