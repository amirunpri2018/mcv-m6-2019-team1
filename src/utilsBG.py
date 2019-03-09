import numpy as np
import utils as ut
# Read frames from video

import cv2 as cv


def getGauss_bg(file_list, D=1 , gt_file = None):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    : git_file , ignoring bbox of foreground when creating the bg model
    :return: mu_bg -  mean, var_bg
    """

# Read the frame from the video and NOT the images

    if D==1:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    # if there is bbox to ignore fron
    if gt_file:
        Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Count the number of Images
    N = len(file_list)
    # get image size

    s = np.shape(cv.imread(file_list[0],Clr_flag))

    # initializing the cumalitive frame matrix
    if D==1:
        A = np.ones((s[0],s[1],N))
        ma = np.ones((s[0],s[1],N),dtype=np.bool)
    for i,image_path in enumerate(file_list, start=0):

        if gt_file:
            frm = ut.frameIdfrom_filename(image_path)

            ma[:,:,i] = ut.getbboxmask(Bbox,frm,(s[0],s[1]))

        #Upload frame
        I = cv.imread(image_path,Clr_flag)

        if D==1:
            A[:,:,i] = I

    mu_bg = A.mean(axis=2)
    std_bg = A.std(axis=2)
    if gt_file:

        for r in range(s[0]):
            for c in range(s[1]):
                p = A[r,c,ma[r,c,:]]
                mu_bg[r,c] = p.mean()
                std_bg[r,c] = p.std()

    return mu_bg,std_bg

def getGauss_bg2(file_list, D=1 , gt_file = None):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    : git_file , ignoring bbox of foreground when creating the bg model
    :return: mu_bg -  mean, var_bg
    """

# Read the frame from the video and NOT the images

    if D==1:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    # if there is bbox to ignore fron
    if gt_file:
        Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Count the number of Images
    N = len(file_list)
    # get image size

    s = np.shape(cv.imread(file_list[0],Clr_flag))
    m0 = np.zeros((s[0],s[1]),dtype=bool)
    # initializing the cumalitive frame matrix
    if D==1:
        A = np.zeros((s[0],s[1]))
        #ma = np.ones((s[0],s[1]),dtype=np.float)
        ma = np.full((s[0],s[1]), float(N))

    # I. Loop to obtain mean
    for i,image_path in enumerate(file_list, start=0):
        if gt_file:
            frm = ut.frameIdfrom_filename(image_path)
            m0 = ut.getbboxmask(Bbox,frm,(s[0],s[1]))
        #Upload frame
        if D==1:
            I = cv.imread(image_path,Clr_flag)
            np.place(I, m0, 0.0)
            ma -= m0
            # Adding frames values
            A+= I
    # Do ma need to be float??
    mu_bg = A/ma

    if D==1:
        A = np.zeros((s[0],s[1]))
    # II. Loop to obtain std
    # sqrt (1/N * sum((x-mu)^2))
    for i,image_path in enumerate(file_list, start=0):
        print i
        if gt_file:
            frm = ut.frameIdfrom_filename(image_path)
            ma0 = ut.getbboxmask(Bbox,frm,(s[0],s[1]))

        #Upload frame
        if D==1:
            I = (cv.imread(image_path,Clr_flag)-mu_bg)**2
            np.place(I, m0, 0.0)
            A+= I

    std_bg = np.sqrt(A/ma)

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
