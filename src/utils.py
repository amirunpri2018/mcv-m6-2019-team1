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

from skimage import exposure
import src.evaluation.evaluation_funcs as evalf
# Local modules

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
    tag = parent.new_tag(tag_name)
    if tag_value is not None:
        tag.string = tag_value
    if tag_attrs is not None and isinstance(tag_attrs, dict):
        tag.attrs = tag_attrs
    parent.annotations.append(tag)


def create_aicity_xml_file(fname, dataframe):
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
        bboxes_tag = soup.find('track')
        # Iterate over bounding boxes and append the attributes to the list
        for child in bboxes_tag.find_all('box'):
            bboxes.append(child.attrs)

    # Return DataFrame
    return pd.DataFrame(bboxes)


<<<<<<< HEAD:code/utils.py
=======
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


def plotOF(img, u, v):
    mag, ang = cv.cartToPolar(u, v)


>>>>>>> 1db897826e621142d5faf239a9b57e8fc8bd50e8:src/utils.py
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
