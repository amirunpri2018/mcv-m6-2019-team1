import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.utils_kalman as ut

#import src

OUTPUT_DIR ='../output'
ROOT_DIR = '../../aic19-track1-mtmc-train/'
INPUT = 'det_yolo3'

# Some constant for the script
TASK = 'task_kalman_multi'
WEEK = 'week5'
SEQ = 'S03'
cameras = ['c011', 'c012', 'c013', 'c014', 'c015']
#CAM2 = 'c011'

SECTION_DIR = os.path.join(ROOT_DIR, 'train', SEQ)
results_dir = os.path.join(OUTPUT_DIR, WEEK, TASK)

# GT_PATHS = [os.path.join(cam, 'gt', 'gt.txt') for cam in CAMERAS]

def read_frames_from_path(path):
    for frame in os.listdir(path):
        frame_path = os.path.join(path, frame)
        yield cv2.imread(frame_path)


def read_frame_number_from_path(path, frame_number):
    frame_path = os.path.join(path, 'frame_%04d.jpg' % frame_number)
    return cv2.imread(frame_path)


def patch_from_img(image, top_left, bottom_right):
    return image[np.int(top_left[0]):np.int(bottom_right[0]), np.int(top_left[1]):np.int(bottom_right[1])]


def match_first_n_descriptors(desc_1, desc_2, n_matches):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(desc_1, desc_2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    dist = [m.distance for m in matches[:n_matches]]

    sqrt_sum = np.sum(np.array(dist)**2) / n_matches
    return sqrt_sum

def find_nearest(array, value):
  idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
  return array[idx]


def main():

    for CAM in cameras:

        FRAMES_DIR1 = os.path.join(SECTION_DIR, CAM, 'frames')
        DET_PATH1 = os.path.join(results_dir, INPUT + SEQ + CAM + '_kalman_predictions.pkl')

        df1 = pd.read_pickle(DET_PATH1)
        df1_sort = df1.sort_values('time_stamp')
        df1_grouped = df1_sort.groupby('time_stamp')

        df1['histogram'] = 0
        df1_new = pd.DataFrame(columns=df1.head(0))

        # Itero sobre cada frame
        for time_stamp, vals in df1_grouped:
            frame_id = vals['img_id'].values[0]
            boxes = vals['boxes']
            frame = read_frame_number_from_path(FRAMES_DIR1, frame_id)
            im_h = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,0]

            # Itero sobre los datos de cada frame
            histograms_per_frame = []
            for b in boxes:
                [xmin, ymin, xmax, ymax] = b
                # Itero sobre cada elemento en el frame
                top_left = (np.int(xmin), np.int(ymin))
                bottom_right = (np.int(xmax), np.int(ymax))
                patch = patch_from_img(im_h, top_left, bottom_right)
                #hist = np.histogram(patch, bins=16)
                hist = np.histogram(patch, bins=np.linespace(0,255,16),density=True)
                histograms_per_frame.append(hist)
                # cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 10)

            vals['histogram'] = histograms_per_frame
            df1_new = df1_new.append(vals, ignore_index=True)

            #plt.imshow(frame)
            #plt.show()
            # plt.imshow(frame)
            # plt.show()
            #cv2.comparehist(hist1, hist2, method=CV_COMP_INTERSECT)

        df1_new.to_pickle(os.path.join(results_dir, INPUT + SEQ + CAM + '_histogram.pkl'))


if __name__ == '__main__':
    main()
