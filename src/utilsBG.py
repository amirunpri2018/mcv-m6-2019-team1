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
    std_bg = A.std(axis=2)

    return mu_bg,std_bg
def foreground_from_GBGmodel(bg_mu,bg_std,I,th =2):
    """
    Apply to any N Gaussian model, (as long the dimensions of I and bg agrees)
    output : Boolien map
        0 Background
        1 Foreground

        For now 8/3 only works for 1D (grayscale)
    """
    s = np.shape(I)
    foreground_map = np.zeros((s[0],s[1]), dtype=bool )

    # centered Image with repect to mu of the Background
    Ic = np.abs(I-bg_mu)
    foreground_map[Ic>=th*(bg_std+2)] = True
    return foreground_map
