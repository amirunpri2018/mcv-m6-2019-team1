# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os
import pickle
import random
import glob

# 3rd party modules
import imageio
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from functools import reduce
from skimage import color
from skimage.measure import label, regionprops
from skimage.transform.integral import integral_image

# Local modules
from evaluation.evaluation_funcs import performance_accumulation_pixel, performance_evaluation_pixel


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_files_from_dir(directory, excl_ext=None):
    """
    Get only files from directory.

    :param directory: Directory path
    :param excl_ext: List with extensions to exclude
    :return: List of files in directory
    """

    logger.debug("Getting files in '{path}'".format(path=os.path.abspath(directory)))

    excl_ext = list() if excl_ext is None else excl_ext

    l = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1] not in excl_ext
    ]
    logger.debug("Retrieving {num_files} files from '{path}'".format(num_files=len(l), path=os.path.abspath(directory)))

    return l


def rgb2hsv(img):
    """
    Convert image from RGB to HSV,

    :param img: Numpy array with the RGB pixel values of the image
    :return: Numpy array with the HSV pixel values of the image
    """

    logger.debug("Converting image from RGB to HSV")
    return color.rgb2hsv(img)


def hsv2rgb(img):
    """
    Convert image from HSV to RGB.

    :param img: Numpy array with the HSV pixel values of the image
    :return: Numpy array with the RGB pixel values of the image
    """

    logger.debug("Converting image from HSV to RGB")
    return color.hsv2rgb(img)


def get_img(folder_dir, img_dir):
    """
    Get numpy array representation of the image.

    :param folder_dir: Folder path
    :param img_dir: Image path
    :return: Numpy array with the RGB pixel values of the image
    """

    img_path = os.path.join(folder_dir, img_dir)
    logger.debug("Getting image '{path}'".format(path=img_path))

    return imageio.imread(img_path)


def get_patch(img, x_min, y_min, x_max, y_max):
    """
    Get patch from an image.

    :param img: Numpy representation of the image
    :param x_min: X min coordinate
    :param y_min: Y min coordinate
    :param x_max: X max coordinate
    :param y_max: Y max coordinate
    :return: Numpy representation of the patch image
    """
    return img[y_min:y_max, x_min:x_max]


def img_name_to_mask_name(filename):
    """
    Get numpy array representation of the test image.

    :param filename: Image name
    :return: Numpy array with the RGB pixel values of the image
    """

    mask_name = 'mask.{filename}'.format(filename=filename.replace('jpg' or 'png', 'txt'))
    logger.debug("'{filename}' converted to '{mask_name}".format(filename=filename, mask_name=mask_name))

    return mask_name


def raw2gt(raw, ext='txt'):
    """
    Convert raw filename format to ground truth format.

    :param raw: Raw filename
    :param ext: Extension of the ground truth file. 'txt' otherwise
    :return: Ground truth filename
    """

    return 'gt.{raw}.{ext}'.format(raw=raw, ext=ext)


def gt2raw(gt, ext='txt'):
    """
    Convert ground truth format to raw filename format.

    :param gt: Ground truth filename
    :param ext: Extension of the ground truth file. 'txt' otherwise
    :return: Raw filename
    """
    return gt.replace('gt.', '').replace('.{ext}'.format(ext=ext), '')


def raw2mask(raw, ext='png'):
    """
    Convert raw filename format to mask format.

    :param raw: Raw filename
    :param ext: Extension of the mask file. 'png' otherwise
    :return: Mask filename
    """

    return 'mask.{raw}.{ext}'.format(raw=raw, ext=ext)


def raw2img(raw, ext='jpg'):
    """
    Convert raw filename format to imageformat.

    :param raw: Raw filename
    :param ext: Extension of the image file. 'jpg' otherwise
    :return: Image filename
    """

    return '{raw}.{ext}'.format(raw=raw, ext=ext)


def gt_to_mask(gt):
    return gt.replace('gt', 'mask').replace('txt', 'png')


def gt_to_img(gt):
    return gt.replace('gt.', '').replace('txt', 'jpg')


def mask_to_gt(gt):
    return gt.replace('mask', 'gt').replace('png', 'txt')


def get_gt_data(dir_path, gt_name):
    f = open(os.path.join(dir_path, gt_name), 'r')
    l = f.readlines()
    f.close()

    return l


def get_unique_values_of_col(df, col):
    return np.unique(df[col].tolist())


def get_n_samples_of_col(df, col, n, val=None):
    if val is None:
        res = df[col]
    else:
        res = df[df[col] == val][col]

    idxs = random.sample(res.index.tolist(), n)

    return [v[0] for v in idxs]


def parse_gt_data(line):
    l = line.strip().split(' ')
    vals = list(map(float, l[:-1]))
    vals.append(l[-1].strip())
    return vals


def int_img(image):
    return integral_image(image)


def sum_over_int_img(int_image, tlx, tly, brx, bry):
    return int_image[bry, brx] - int_image[tly, brx] - int_image[bry, tlx] + int_image[tly, tlx]


def threshold_image(img, ths, channel=0):
    """
    Get the mask of a image for the pixels that are between a set threshold values.

    :param img: Numpy array representation of the image
    :param ths: List of tuples of thresholds
    :param channel: Channel of interest of the image. Default 0
    :return: Numpy array with the mask for values between threshold values
    """

    logger.debug("Getting image pixels of channel {channel} that are between the thresholds".format(channel=channel))
    # Create a mask with the same shape of the image filled with 'False'
    mask = np.full(img.shape[:2], False)
    # Get the channel of the image
    c = img[..., channel]

    # Iterate over thresholds
    for th in ths:
        # Add the values between the thresholds as 'True' values to the mask
        mask += np.logical_and(th[0] <= c, c <= th[1])

    return mask


def save_image(img, directory, name, ext='png'):
    """
    Save a numpy array as an image.

    :param img: Numpy array with the pixel values of the image
    :param directory: Folder where the image will be saved
    :param name: Filename of the image
    :param ext: Extension of de image file
    :return: Filename of the image and folder where it was saved
    """

    # Get filename without extension and the new one
    filename = '.'.join(name.split('.')[:-1])
    filename = '{name}.{ext}'.format(name=filename, ext=ext)

    if not os.path.exists(directory):
        logger.debug("{directory} does not exist".format(directory=directory))
        os.mkdir(directory)

    img_path = os.path.join(directory, filename)

    logger.debug("Saving image in {path}".format(path=img_path))

    # Save image
    imageio.imwrite(img_path, img)

    return filename, directory


def confusion_matrix(results_dir, masks_dir):
    """
    Calculate confusion matrix.

    :param results_dir: Directory with calculated masks
    :param masks_dir: Ground truth masks
    :return: Confusion matrix
    """

    # Getting calculated masks
    #result_imgs = get_files_from_dir(results_dir)
    result_imgs = glob.glob(results_dir + "/*.png")

    # List with values TP, FP, FN, TN
    tf_values = np.zeros(4)

    # Iterate over image paths
    for img_path in result_imgs:
        print(img_path)
        image_name = img_path.split("/")[-1]
        # Convert image path to mask path
        # mask_path = img_name_to_mask_name(img_path) if not img_path.startswith('mask.') else img_path
        mask_path = os.path.join(masks_dir, image_name)
        print(mask_path)

        # Compute perfomance measures
        tf_val = np.array(performance_accumulation_pixel(
            #get_img(results_dir, img_path),
            #get_img(masks_dir, mask_path)
            imageio.imread(img_path),
            imageio.imread(mask_path)
        ))

        # Add them up
        tf_values += tf_val

    return tf_values.tolist()


def print_confusion_matrix(values):
    """
    Print a human-readable representation of a confusion matrix.

    :param values: Values of the confusion matrix
    :return: Nothing
    """

    # Reshape matrix values
    values = np.array(values).reshape((2, 2))

    return np.array2string(values) + '\n'


def print_pixel_metrics(values):
    l = 'Precision  : %.3f\n' % values[0]
    l += 'Accuracy   : %.3f\n' % values[1]
    l += 'Specificity: %.3f\n' % values[2]
    l += 'Sensitivity: %.3f\n' % values[3]
    l += 'F1 Score   : %.3f\n' % (2 * values[0] * values[2] / (values[0] + values[2]))

    return l


def print_window_metrics(values):
    l = 'Precision  : %.3f\n' % values[0]
    l += 'Accuracy   : %.3f\n' % values[1]
    l += 'Specificity: %.3f\n' % values[2]
    l += 'Sensitivity: %.3f\n' % values[3]
    l += 'F1 Score   : %.3f\n' % (2 * values[0] * values[2] / (values[0] + values[2]))

    return l


def get_data(data_dir, gt_dir=None, mask_dir=None):
    if gt_dir is None:
        columns = ['img_file']

        l = list(map(lambda x: '.'.join(x.split('.')[:-1]), get_files_from_dir(data_dir)))
    else:
        columns = ['img_file', 'type', 'width', 'height', 'bbox_area', 'form_factor',
                   'tly', 'tlx', 'bry', 'brx', 'mask_area', 'filling_ratio', 'mask']

        gts = os.listdir(gt_dir)

        l = list()
        for gt in gts:
            raw_name = gt2raw(gt)
            mask_path = gt_to_mask(gt)
            mask_img = get_img(mask_dir, mask_path)
            # orig_img = get_train_img(gt_to_img(gt))

            for gt_datum in get_gt_data(gt_dir, gt):
                tly, tlx, bry, brx, signal_type = parse_gt_data(gt_datum)

                d = dict()

                w = brx - tlx
                h = bry - tly

                d['img_file'] = raw_name
                d['type'] = signal_type.strip()
                d['width'] = w
                d['height'] = h
                d['bbox_area'] = w * h
                d['form_factor'] = w / h

                d['tly'] = round(tly)
                d['tlx'] = round(tlx)
                d['bry'] = round(bry)
                d['brx'] = round(brx)

                mask_patch = get_patch(mask_img, d['tlx'], d['tly'], d['brx'], d['bry'])
                mask_area = np.count_nonzero(mask_patch)
                d['mask_area'] = mask_area
                d['filling_ratio'] = mask_area / d['bbox_area']
                d['mask'] = mask_patch != 0

                l.append(list(d.values()))

    return pd.DataFrame(l, columns=columns)


def split_data_by(data, key=None, train_size=0.7):
    _keys = list(data.values())[0][0].keys()

    if key is None:
        keys = list(data.keys())
        train_sz = round(len(keys) * train_size)

    elif key in _keys:
        pass
    else:
        raise ValueError('"{key}" key not in data'.format(key=key))


def split_data_by_type(data, train_size=0.7):
    unique_data = data.drop_duplicates('gt_file', keep='first')

    print(unique_data)
    pass


def bbox_to_pkl(bboxes, fname, folder=''):
    if not isinstance(bboxes, list):
        raise ValueError('bboxes variable must be a list')

    fname = fname if fname.endswith('.pkl') else '{fname}.pkl'.format(fname=fname)
    path = os.path.join(folder, fname)

    with open(path, 'wb') as f:
        pickle.dump(bboxes, f)


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
        return []

    # If the bounding boxes integers, convert them to floats
    # This is important since we'll be doing a bunch of divisions
    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes
    # list
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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap_thresh < overlap)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return bboxes[pick].astype("int")


def bboxes_to_file(bboxes, fname, directory, sign_types=None):
    if sign_types is not None and len(bboxes) != len(sign_types):
        raise ValueError('bboxes and sign_types have different sizes')

    if not os.path.exists(directory):
        os.mkdir(directory)

    fpath = os.path.join(directory, fname)
    with open(fpath, 'w+') as f:
        for idx, bbox in enumerate(bboxes):
            f.write('{tlx} {tly} {brx} {bry} '.format(tlx=bbox[0], tly=bbox[1], brx=bbox[2], bry=bbox[3]))
            f.write('{signal_type}\n'.format(signal_type=sign_types[idx] if sign_types is not None else ''))


def merge_masks(masks):
    return 1 * reduce(np.bitwise_or, masks)


def connected_components(mask0, area_min=None, area_max=None, ff_min=None, ff_max=None, fr_min=None, plot=False):
    """
    :param mask0: Incoming mask (2D array)
    :param area_min: Min area allowed for bbox
    :param area_max: Max area allowed for bbox
    :param ff_min: Min form factor allowed for bbox
    :param ff_max: Max form factor allowed for bbox
    :param fr_min: Min filling ratio allowed for bbox
    :param plot: If 'true' plots mask + selected bboxes
    """

    label_image = label(mask0)
    bbox_list = []

    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(mask0)

    for region in regionprops(label_image):

        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        form_factor = w / h
        filling_ratio = region.filled_area / region.bbox_area

        # Filter by area:
        if area_min is not None and area_max is not None:
            if area_min <= region.bbox_area <= area_max:
                minr, minc, maxr, maxc = region.bbox
            else:
                del (minr, minc, maxr, maxc)
                continue

        # Filter by form factor:
        if ff_min is not None and ff_max is not None:
            if ff_min < form_factor < ff_max:
                minr, minc, maxr, maxc = region.bbox
            else:
                del (minr, minc, maxr, maxc)
                continue

        # Filter by filling ratio:
        if fr_min is not None:
            if filling_ratio > fr_min:
                minr, minc, maxr, maxc = region.bbox
            else:
                del (minr, minc, maxr, maxc)
                continue

        bbox_list.append([minc, minr, maxc, maxr])

        if plot:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    return bbox_list


def text2file(text, fname, dir):
    """
    Save string into file.

    :param text: String to save
    :param fname: Filename of the text file
    :param dir: Directory where the text file will be placed
    :return: Path of the text file
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(dir):
        os.mkdir(dir)

    # Append 'txt' sufix if the filename doesn't end with 'txt
    if not fname.endswith('.txt'):
        fname += '.txt'

    # If text is not a list, convert it to iterate over lines
    if not isinstance(text, list):
        text = [text]

    fpath = os.path.join(dir, fname)
    with open(fpath, 'a') as f:
        for line in text:
            f.write(line)

    return fpath


def bbox2evalformat(bbox):
    """
    Convert bounding box representation to the one used for evaluation.

        [tly, tlx, bry, brx] => [x, y, w, h]

    :param bbox: Bounding box
    :return: Evaluation representation of the boundin box
    """

    tly, tlx, bry, brx = bbox
    x, y = tlx, tly
    w, h = brx - tlx, bry - tly

    return x, y, w, h


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


##########################################################################


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
