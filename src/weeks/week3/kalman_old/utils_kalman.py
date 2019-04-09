import json
import logging

# 3rd party modules

import numpy as np
import pandas as pd

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_bboxes_from_MOTChallenge_for_map(fname):
    """
    Read GT as format required in map_metrics.py

    Get the Bboxes from the txt files
    MOTChallengr format [frame,ID,left,top,width,height,1,-1,-1,-1]
     {'ymax': 84.0, 'frame': 90, 'track_id': 2, 'xmax': 672.0, 'xmin': 578.0, 'ymin': 43.0, 'occlusion': 1}
    fname: is the path to the txt file
    :returns: Pandas DataFrame with the data
    """
    f = open(fname, "r")
    BBox_list = list()

    for line in f:
        data = line.split(',')
        xmax = float(data[2]) + float(data[4])
        ymax = float(data[3]) + float(data[5])

        BBox_list.append({'img_id': int(data[0]),
                          'track_id': int(data[1]),
                          'boxes': [float(data[2]), float(data[3]), xmax, ymax],
                          'occlusion': 1,
                          'conf': float(data[6])})

    return pd.DataFrame(BBox_list)


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
        BBox_list.append({'frame':int(data[0]),'track_id':int(data[1]), 'xmin':float(data[2]), 'ymin':float(data[3]), 'xmax':xmax, 'ymax':ymax,'occlusion': 1,'conf' :float(data[6])})

    return pd.DataFrame(BBox_list)





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

    # Save DataFrame if necessary
    if save_in is not None:
        df.to_pickle(save_in)

    # Return DataFrame
    return df



def locate_panda_feature(pd_dataframe, key, value):
    """
    Group pandas dataframe by selected key
    """
    grouped_pd = pd_dataframe.groupby(key).agg(lambda x: list(x))

    return grouped_pd.loc[value]


def kalman_out_to_pandas(out_kalman):
    """
    :param out_kalman: Output from kalman tracking
    :returns: Panda dataframe with format 'frame', 'ymin', 'xmin', 'ymax', 'xmax', 'track_id'
    """

    vals = list()

    for frame_data in out_kalman:

        frame = frame_data[0]
        frame_vals = [frame]

        for track in frame_data[1]:
            ymin, xmin, ymax, xmax, track_id = track

            # score = 1
            score = np.random.uniform(0, 1)

            frame_vals = [frame, ymin, xmin, ymax, xmax, track_id, score]

            vals.append(frame_vals)

    df_kalman = pd.DataFrame(vals, columns=['frame', 'ymin', 'xmin', 'ymax', 'xmax', 'track_id', 'score'])

    return df_kalman



def kalman_out_to_pandas_for_map(out_kalman):
    """
    Prepair dictionary for map_metrics.py
    :param out_kalman: Output from kalman tracking
    :returns: Panda dataframe with format 'img_id', 'boxes', 'track_id', 'scores'
    """

    vals = list()

    for frame_data in out_kalman:

        img_id = frame_data[0]
        #frame_vals = [frame]

        for track in frame_data[1]:

            xmin, ymin, xmax, ymax, track_id = track

            boxes = [xmin, ymin, xmax, ymax]

            scores = np.random.uniform(low=0.8, high=1.0)

            frame_vals = [img_id, boxes, track_id, scores]

            vals.append(frame_vals)

    df_kalman = pd.DataFrame(vals, columns=['img_id', 'boxes', 'track_id', 'scores'])

    return df_kalman


def get_bboxes_from_MOTChallenge_for_map(fname):
    """
    Read GT as format required in map_metrics.py

    Get the Bboxes from the txt files
    MOTChallengr format [frame,ID,left,top,width,height,1,-1,-1,-1]
     {'ymax': 84.0, 'frame': 90, 'track_id': 2, 'xmax': 672.0, 'xmin': 578.0, 'ymin': 43.0, 'occlusion': 1}
    fname: is the path to the txt file
    :returns: Pandas DataFrame with the data
    """
    f = open(fname, "r")
    BBox_list = list()

    for line in f:
        data = line.split(',')
        xmax = float(data[2]) + float(data[4])
        ymax = float(data[3]) + float(data[5])

        BBox_list.append({'img_id': int(data[0]),
                          'track_id': int(data[1]),
                          'boxes': [float(data[2]), float(data[3]), xmax, ymax],
                          'occlusion': 1,
                          'conf': float(data[6])})

    return pd.DataFrame(BBox_list)


def get_bboxes_from_xml_for_map(gt_panda):
    """
    Prepair dictionary for map_metrics.py
    :param out_kalman: Output from kalman tracking
    :returns: Panda dataframe with format 'img_id', 'boxes', 'track_id'
    """

    vals = list()

    for frame_data in gt_panda:

        img_id = frame_data[0]
        frame_vals = [frame]

        boxes = []

        for track in frame_data[1]:
            ymin, xmin, ymax, xmax, track_id = track

            boxes.append([xmin, ymin, xmax, ymax])

            scores = np.random.uniform(low=0.8, high=1.0)

        frame_vals = [img_id, boxes, track_id, scores]

        vals.append(frame_vals)

    df_kalman = pd.DataFrame(vals, columns=['img_id', 'boxes', 'track_id', 'scores'])

    return df_kalman


####################################################

def panda_to_json_predicted(predicted_panda):
    """
    Transform predicted boxes panda to json file used in map_metrics.py
    """
    d_pred = {}
    for name, group in predicted_panda.groupby('img_id'):
        boxes = []
        scores = []
        for idx in range(len(group)):
            boxes.append(group.iloc[idx].boxes)

            scores.append(group.iloc[idx].scores)
        d_pred[group.iloc[idx].img_id] = {'boxes': boxes, 'scores': scores}

    return d_pred


def panda_to_json_gt(gt_panda):
    """
    Transform gt boxes panda to json file used in map_metrics.py
    """
    d_gt = {}
    for name, group in gt_panda.groupby('img_id'):
        boxes = []
        for idx in range(len(group)):
            boxes.append(group.iloc[idx].boxes)
        d_gt[group.iloc[idx].img_id] = {'boxes': boxes}

    return d_gt


def save_json(panda_file, filename):
    with open(filename, 'w') as fp:
        json.dump(panda_file, fp)


def save_pkl(panda_file, filename):
    panda_file.to_pickle(filename)