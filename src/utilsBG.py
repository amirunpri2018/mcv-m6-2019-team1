import numpy as np
import utils as ut
# Read frames from video

import cv2 as cv

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.animation as manimation
# For visulization
import matplotlib.pyplot as plt
import scipy.stats as stats

# mu = 0
# variance = 1
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))
# plt.show()
# fig = plt.figure()
# l, = plt.plot([], [], 'k-o')
#
#
# with writer.saving(fig, "writer_test.mp4", 100):
#     for i in range(100):
#         x0 += 0.1 * np.random.randn()
#         y0 += 0.1 * np.random.randn()
#         l.set_data(x0, y0)
#         writer.grab_frame()


def adaptive_BG(bg_mu,bg_std,I,p=0.5,th=2,D=1):
        """
        Computes Gaussian Background model.

        :file_list: list of all images used to estimate the bg model
        :D: 1 - Greyscale
            3 - RGB/HSV/any other color space
        : BG_model  - size (M,N,D)
        : current_frame : has to fit the model , size(M,N,D) - matrix (not a file path)
        : th - alpha parameters - the descision threshold
        : p : [0,1]- the weight between the past model and the current frame
        """
        fg_map = foreground_from_GBGmodel(bg_mu,bg_std,I,th)
        bg_var = np.sqrt(bg_std)

        # Loop on every pixel that was found as Background
        bg_list = np.where(~fg_map)
        for bg_pix in bg_list:
            bg_mu[bg_pix[0]][bg_pix[1]] = p*I[bg_pix[0]][bg_pix[1]]+(1-p)*bg_mu[bg_pix[0]][bg_pix[1]]
            bg_var[bg_pix[0]][bg_pix[1]] = p*(I[bg_pix[0]][bg_pix[1]]-bg_mu[bg_pix[0]][bg_pix[1]])**2+(1-p)*bg_var[bg_pix[0]][bg_pix[1]]

        bg_std = np.sqrt(bg_var)
        return bg_mu,bg_std


def getGauss_bg(file_list, D=1 ,color_space = None,color_channels=None, gt_file = None):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    : git_file , ignoring bbox of foreground when creating the bg model
    :return: mu_bg -  mean, var_bg - chose to work with var and not std var = std^2 !!
    """

    # if D==1:
    #     Clr_flag = cv.IMREAD_GRAYSCALE
    # else :
    #     Clr_flag = cv.IMREAD_COLOR


    # if there is bbox to ignore fron
    if gt_file:
        Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Count the number of Images

    # get image size
    N = len(file_list)

    s = np.shape(ut.getImg_D(file_list[0],D,color_space,color_channels))
    m0 = np.zeros((s[0],s[1],D),dtype=bool)

    # initializing the cumalitive frame matrix
    A = np.zeros((s[0],s[1],D))

    ma = np.full((s[0],s[1],D), float(N))

    # I. Loop to obtain mean


    #for j in range(n):
        #update_figure(n)         #moviewriter.grab_frame()
    for i,image_path in enumerate(file_list, start=0):
        # Get frame Number
        frm = ut.frameIdfrom_filename(image_path)
        #Upload frame
        I = ut.getImg_D(image_path,D,color_space,color_channels)

        if gt_file:
            m0,_ = ut.getbboxmask(Bbox,frm,(s[0],s[1]))
            m0 = np.repeat(m0[:, :, np.newaxis], D, axis=2)

            # Read the frame from the video and NOT the images

        np.place(I, m0, 0.0)
        ma -= m0
        # Adding frames values
        A+= I

    mu_bg = A/ma
    m0 = np.zeros((s[0],s[1],D),dtype=bool)


    A = np.zeros((s[0],s[1],D))
    #A = np.zeros((s[0],s[1]))
    # II. Loop to obtain std
    # sqrt (1/N * sum((x-mu)^2))
    for i,image_path in enumerate(file_list, start=0):
        # get frame number
        frm = ut.frameIdfrom_filename(image_path)
        #Upload frame
        I = ut.getImg_D(image_path,D,color_space,color_channels)

        Ivar = (I-mu_bg)**2

        #print i
        if gt_file:

            m0,_ = ut.getbboxmask(Bbox,frm,(s[0],s[1]))
            m0 = np.repeat(m0[:, :, np.newaxis], D, axis=2)


        np.place(Ivar, m0, 0.0)
        A+= Ivar

    #var_bg = A/ma
    std_bg = np.sqrt(A/ma)

    return mu_bg,std_bg



def foreground_from_GBGmodel(bg_mu,bg_std,I,th =2):
    """
    Apply to any N Gaussian model, (as long the dimensions of I and bg agrees)
    output : Boolien map
        0 Background
        1 Foreground

        For now 8/3 only works for 1D (grayscale)

        th = 1:
        (confedence interval for pixel to belong to the backgroud)
        - 1 std ==  68% CI
        - 2 std ==  95% CI
        - 3 std ==  99.7 % CI

    """
    s = np.shape(I)
    #fg_map = np.zeros((s[0],s[1]), dtype=bool )
    fg_map = np.zeros(s, dtype=bool )
    # centered Image with repect to mu of the Background
    Ic = np.abs(I-bg_mu)

    for d in range(s[2]):
        fg_map[Ic[...,d]>=th*(bg_std[...,d]+2),d] = True

    #np.any(fg_map,axis=2)
    return np.any(fg_map,axis=2)


# State of the art background subtraction:


def background_subtractor_MOG(cap):

    """
    cap = cv.VideoCapture('vdo.avi')
    """

    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv.imshow('frame',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()


def background_subtractor_MOG2(cap):

    """
    cap = cv.VideoCapture('vdo.avi')
    """

    fgbg = cv.createBackgroundSubtractorMOG2()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv.imshow('frame',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()


def background_subtractor_GMG(cap):

    """
    cap = cv.VideoCapture('vdo.avi')

    Warning: first frames will be black
    """

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        cv.imshow('frame',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()


def background_subtractor_LSBP(cap):

    """
    cap = cv.VideoCapture('vdo.avi')
    """

    fgbg = cv.bgsegm.createBackgroundSubtractorLSBP()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv.imshow('frame',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
