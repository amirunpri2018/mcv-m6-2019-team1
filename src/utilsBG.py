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


def getGauss_bg(file_list, D=1 , gt_file = None):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    : git_file , ignoring bbox of foreground when creating the bg model
    :return: mu_bg -  mean, var_bg - chose to work with var and not std var = std^2 !!
    """

    if D==1:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    # if there is bbox to ignore fron
    if gt_file:
        Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Count the number of Images

    # get image size
    N = len(file_list)

    s = np.shape(cv.imread(file_list[0],Clr_flag))
    #m0 = np.zeros((s[0],s[1],D),dtype=bool)
    m0 = np.zeros((s[0],s[1]),dtype=bool)
    # initializing the cumalitive frame matrix
    A = np.zeros((s[0],s[1]))
    #A = np.zeros((s[0],s[1],D))
    #B = np.zeros((s[0],s[1]))
    #G = np.zeros((s[0],s[1]))
    #R = np.zeros((s[0],s[1]))
    #ma = np.full((s[0],s[1],D), float(N))
    ma = np.full((s[0],s[1]), float(N))

    # I. Loop to obtain mean


    #for j in range(n):
        #update_figure(n)         #moviewriter.grab_frame()
    for i,image_path in enumerate(file_list, start=0):
        # Get frame Number
        frm = ut.frameIdfrom_filename(image_path)
        #Upload frame
        I= cv.imread(image_path,Clr_flag)
        #print(np.shape(I))

        if gt_file:
            m0,_ = ut.getbboxmask(Bbox,frm,(s[0],s[1]))
            #m0 = np.repmat(m0,1,1,D)

        if D==1:
            # Read the frame from the video and NOT the images

            np.place(I, m0, 0.0)
            ma -= m0
            # Adding frames values
            A+= I
        if D==3:
            # Read the frame from the video and NOT the images
            bgr = cv.split(I)
            b = bgr[0]
            g = bgr[1]
            r = bgr[2]
            np.place(b, m0, 0.0)
            np.place(g, m0, 0.0)
            np.place(r, m0, 0.0)
            ma -= m0
            # Adding frames values
            B += b
            G += g
            R += r
            #A+= I

    # Do ma need to be float??
    if D==1:
        mu_bg = A/ma
    else:
        B = B/ma
        R = R/ma
        G =G/ma
        mu_bg = cv.merge((B,G,R))

    #A = np.zeros((s[0],s[1],D))
    A = np.zeros((s[0],s[1]))
    # II. Loop to obtain std
    # sqrt (1/N * sum((x-mu)^2))
    for i,image_path in enumerate(file_list, start=0):
        # get frame number
        frm = ut.frameIdfrom_filename(image_path)
        #Upload frame
        I = (cv.imread(image_path,Clr_flag)-mu_bg)**2
        #print i
        if gt_file:

            ma0,_ = ut.getbboxmask(Bbox,frm,(s[0],s[1]))
            m0 = np.repmat(m0,1,1,D)

        if D==1 or D==3:

            np.place(I, m0, 0.0)
            A+= I

    #var_bg = A/ma
    std_bg = np.sqrt(A/ma)

    return mu_bg,std_bg

def getGauss_bg2(file_list, D=1 , gt_file = None, output_dir = None):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    : git_file , ignoring bbox of foreground when creating the bg model
    :return: mu_bg -  mean, var_bg - chose to work with var and not std var = std^2 !!
    """


# Create aa video of constructing the background MODEL
    print(output_dir)


    if D==1:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    # if there is bbox to ignore fron
    if gt_file:
        Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Count the number of Images

    # get image size
    N = len(file_list)

    s = np.shape(cv.imread(file_list[0],Clr_flag))
    #m0 = np.zeros((s[0],s[1],D),dtype=bool)
    m0 = np.zeros((s[0],s[1]),dtype=bool)
    # initializing the cumalitive frame matrix

    A = np.zeros((s[0],s[1],D))
    B = np.zeros((s[0],s[1]))
    G = np.zeros((s[0],s[1]))
    R = np.zeros((s[0],s[1]))
    #ma = np.full((s[0],s[1],D), float(N))
    ma = np.full((s[0],s[1]), float(N))
    #fig = plt.figure(1)

    if output_dir:
        #FFMpegWriter = manimation.writers['ffmpeg']
        #writer = FFMpegWriter(fps=20, metadata=dict(artist='Team1',title='BG model'))

        fig_ani = plt.figure(2)
        ax1_ani = plt.subplot(221) # mu
        ax2_ani = plt.subplot(222) # std
        ax3_ani = plt.subplot(223) # mask
        ax4_ani = plt.subplot(224) # mask
        ax1_ani.imshow(A,cmap='gray')
        ax2_ani.imshow(A,cmap='gray')
        ax3_ani.imshow(ma,cmap='gray',vmin=0,vmax=N)
        ax4_ani.imshow(m0,cmap='gray')

        #writer.saving(fig_ani, vid_file, dpi=100)
    # I. Loop to obtain mean


    #for j in range(n):
        #update_figure(n)         #moviewriter.grab_frame()
    for i,image_path in enumerate(file_list, start=0):
        # Get frame Number
        frm = ut.frameIdfrom_filename(image_path)
        #Upload frame
        I= cv.imread(image_path,Clr_flag)

        if gt_file:
            m0,_ = ut.getbboxmask(Bbox,frm,(s[0],s[1]))
            #m0 = np.repmat(m0,1,1,D)

        if D==1:
            # Read the frame from the video and NOT the images

            if output_dir:
                ax2_ani.cla()
                ax4_ani.cla()
                ax2_ani.imshow(I,cmap='gray',vmin=0,vmax=255)
                ax4_ani.imshow(m0,cmap='gray',vmin=0,vmax=1)
                ax4_ani.set_title("Current mask")
                ax2_ani.set_title("frame {}".format(frm))
                plt.pause(0.01)

            np.place(I, m0, 0.0)
            ma -= m0
            # Adding frames values
            ax3_ani.cla()
            ax3_ani.imshow(np.abs(A/(i)-I),vmin=0,vmax=255)
            A+= I

            if output_dir:
                ax1_ani.cla()
                #ax3_ani.cla()
                ax1_ani.imshow(A/(i+1),cmap='gray',vmin=0,vmax=255)
                #ax3_ani.imshow(ma,cmap='gray',vmin=0,vmax=N)

                ax1_ani.set_title("Cumulative frames")
                fig_ani.savefig(output_dir+'frm'+str(frm)+'.png')

        if D==3:
            # Read the frame from the video and NOT the images
            bgr = cv.split(I)
            b = bgr[0]
            g = bgr[1]
            r = bgr[2]
            if output_dir:
                ax2_ani.cla()
                ax4_ani.cla()
                ax2_ani.imshow(b,cmap='gray',vmin=0,vmax=255)
                ax4_ani.imshow(m0,cmap='gray',vmin=0,vmax=1)
                ax4_ani.set_title("Current mask")
                ax2_ani.set_title("frame {}".format(frm))
                plt.pause(0.01)



            np.place(b, m0, 0.0)
            np.place(g, m0, 0.0)
            np.place(r, m0, 0.0)
            ma -= m0
            # Adding frames values
            ax3_ani.cla()
            ax3_ani.imshow(np.abs(B/i-b),vmin=0,vmax=255)
            B += b
            G += g
            R += r
            #A+= I

            if output_dir:
                ax1_ani.cla()
                #ax3_ani.cla()
                #ax1_ani.imshow(B/(i+1),cmap='gray',vmin=0,vmax=255)
                ax1_ani.imshow(B/float(i+1),cmap='gray',vmin=0,vmax=255)
                #ax3_ani.imshow(ma,cmap='gray',vmin=0,vmax=N)

                ax1_ani.set_title("Cumulative frames")
                fig_ani.savefig(output_dir+'frm'+str(frm)+'.png')


        #writer.grab_frame()
                            #ax1_ani.cla()
    # Do ma need to be float??
    if D==1:
        mu_bg = A/ma
    else:
        B = B/ma
        R = R/ma
        G =G/ma
        mu_bg = cv.merge((B,G,R))

    if output_dir:
        ax1_ani.cla()
        ax1_ani.imshow(mu_bg,cmap='gray')
        ax1_ani.set_title("Mean BG, over {} frames".format(frm))


    A = np.zeros((s[0],s[1],D))
    # II. Loop to obtain std
    # sqrt (1/N * sum((x-mu)^2))
    for i,image_path in enumerate(file_list, start=0):
        # get frame number
        frm = ut.frameIdfrom_filename(image_path)
        #Upload frame
        I = (cv.imread(image_path,Clr_flag)-mu_bg)**2
        #print i
        if gt_file:

            ma0,_= ut.getbboxmask(Bbox,frm,(s[0],s[1]))
            m0 = np.repmat(m0,1,1,D)

        if D==1 or D==3:

            if output_dir:
                ax2_ani.cla()
                ax2_ani.imshow(I,cmap='gray')
                ax2_ani.set_title("frame #{}".format(frm))
                ax4_ani.cla()
                ax4_ani.imshow(m0,cmap='gray',vmin=0,vmax=1)
                ax4_ani.set_title("Current mask")
            np.place(I, m0, 0.0)
            A+= I
            if output_dir:
                ax3_ani.cla()
                ax3_ani.imshow(A/i,cmap='gray')
                ax3_ani.set_title("Cumulative STD")
                fig_ani.savefig(output_dir+'frm'+str(frm)+'.png')


    #var_bg = A/ma
    std_bg = np.sqrt(A/ma)
    #ani.save('HeroinOverdosesJumpy.mp4', writer=writer)
    if output_dir:
        ax3_ani.cla()
        ax3_ani.imshow(var_bg,cmap='gray')
        #ax3_ani.imshow(std_bg,cmap='gray')
        ax3_ani.set_title("Std BG, over {} frames".format(frm))
        fig_ani.savefig(output_dir+'final.png')
        plt.close(fig_ani)
    return mu_bg,std_bg
    #return mu_bg,var_bg



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
    fg_map = np.zeros((s[0],s[1]), dtype=bool )
    # centered Image with repect to mu of the Background
    Ic = np.abs(I-bg_mu)
    fg_map[Ic>=th*(bg_std+2)] = True
    return fg_map


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
