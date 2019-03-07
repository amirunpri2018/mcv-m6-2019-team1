import numpy as np
import utils as ut

import cv2 as cv


def getGauss_bg(file_list, D=1):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    :return: mu_bg -  mean, var_bg
    """

    if D==1:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    # Count the number of Images
    N = len(file_list)
    # get image size

    s = np.shape(cv.imread(file_list[0],Clr_flag))
    print([s[0],s[1],N])
    # initializing the cumalitive frame matrix
    if D==1:
        A = np.ones((s[0],s[1],N))

    for i,image_path in enumerate(file_list, start=0):

        I = cv.imread(image_path,Clr_flag)
        if D==1:
            A[:,:,i] = I
        #I = get_img(file_list[i])

    mu_bg = A.mean(axis=2)
    var_bg = A.var(axis=2)

    return mu_bg,var_bg
