import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src

SECTION_DIR = os.path.join(src.AICHALLENGE_DIR, 'train', 'S01')
# CAMERAS = ['c001', 'c002']
CAMERAS = ['c001']
FRAMES_DIRS = [os.path.join(SECTION_DIR, cam, 'frames') for cam in CAMERAS]
GT_PATHS = [os.path.join(cam, 'gt', 'gt.txt') for cam in CAMERAS]


def read_frames_from_path(path):
    for frame in os.listdir(path):
        frame_path = os.path.join(path, frame)
        yield cv2.imread(frame_path)


def read_frame_number_from_path(path, frame_number):
    frame_path = os.path.join(path, 'frame_%04d.jpg' % frame_number)
    return cv2.imread(frame_path)


def patch_from_img(image, top_left, bottom_right):
    return image[top_left[1]:bottom_right[1], top_left[1]:bottom_right[1]]


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


def main():
    dfs = dict()
    for camera, gt in zip(CAMERAS, GT_PATHS):
        df = pd.read_csv(os.path.join(SECTION_DIR, gt), sep=',',
                         names=['frame', 'ID', 'left', 'top', 'width',
                                'height', 'drop1', 'drop2', 'drop3', 'drop4'])

        df = df.drop(columns=['drop1', 'drop2', 'drop3', 'drop4'])
        df = df.assign(kpts=None, desc=None)
        df = df.sort_values('frame')
        dfs[camera] = df.copy()

    max_num_of_frames = max([len(os.listdir(folder)) for folder in FRAMES_DIRS])
    sift = cv2.xfeatures2d.SIFT_create()

    descs_prev = list()
    # descs_prev = np.array([])
    # Itero sobre cada frame
    for frame_id in range(104, max_num_of_frames):

        frame = read_frame_number_from_path(FRAMES_DIRS[0], frame_id)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        descs_next = list()
        # descs_next = np.array([])
        # Itero sobre los datos de cada del frame
        for idx, row in df[df.frame == frame_id].iterrows():
            # Itero sobre cada elemento en el frame
            top_left = (row.left, row.top)
            bottom_right = (row.left + row.width, row.top + row.height)

            patch = patch_from_img(gray, top_left, bottom_right)
            cv2.imshow(patch)

            _, desc = sift.detectAndCompute(patch, None)
            descs_next.append(desc)
            # descs_next = np.append(descs_next, desc)
            # print(top_left, bottom_right)
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 10)
        # plt.imshow(frame)
        # plt.show()

        if len(descs_prev) != 0:
            match_first_n_descriptors(descs_prev, descs_next, 10)

        descs_prev = descs_next[:]
        # descs_prev = descs_next.copy()


if __name__ == '__main__':
    main()
#
#
#
# img = cv2.imread('home.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
#
# img = cv2.drawKeypoints(gray, kp)
# cv2.imshow(img)

# # TODO
# - Buscar imagen con varios bboxes y comparar a ver si acierta
# - Usar secuencia de video y buscar matches
# - Acomodar DF para que anote si esta matcheado o no
# - Buscar por timestamp
# - Generalizar para mas de 1 camara
