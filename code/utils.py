import xml
import os
import numpy as np


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


def create_or_destroy_bboxes(bboxes, prob=0.5):
    """
    Create or destroy bounding boxes based on probability value.

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
    # If there is only one bounding box, change to a list
    if not (isinstance(bboxes, list) or isinstance(bboxes, np.array)):
        bboxes = list(bboxes)

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
    for bbox in bboxes:
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
    return bboxes


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
            score += num_hits / (i+1.0)

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


def xml_pascal_to_aicity(xml_file, folder=None):
    xml_output = None
    e = xml.etree.ElementTree.parse(xml_file).getroot()



def readOF(OFdir,filename):

    # Sequance 1
    OF_path = os.path.join(OFdir ,filename)
    OF = cv.imread(gt1_path,-1)
    u = (OF[:,:,1].ravel()-2**15) / 64.0
    v = (OF[:,:,2].ravel()-2**15) / 64.0
    valid_OF = OF[:,:,0].ravel()

    u = np.multiply(u,valid_OF)
    v = np.multiply(v,valid_OF)
    return u ,v

def plotOF(img, u, v):
    mag, ang = cv.cartToPolar(u,v)
