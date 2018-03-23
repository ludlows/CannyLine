"""
implement canny line
"""

import cv2
import numpy as np

class MetaLine(object):
    """
    pass
    """
    def __init__(self):
        """
        default parameters
        """
        self.visual_meaning_grad = 70
        self.p = 0.125
        self.sigma = 1.0
        self.thresh_angle = 0.0
    
    def getInfo(self, origin_img, gauss_sigma, gauss_half_size, p):
        """
        update paramter and bind them with self
        """
        gray_level_num = 255
        aperture_size = 3
        angle_per = np.pi / 8.0
        gauss_noise = 1.3333;
        threshold_grad_low = gauss_noise

        self.num_row, self.num_col = origin_img.shape
        n4 = np.square(self.num_row * self.num_row)

        pixels_num = self.num_row * self.num_col
        # meaningful length
        self.meaningful_len = int(2.0 * np.log(pixels_num) / np.log(8.0) + 0.5)
        self.thresh_angle = 2 * np.arctan(2.0 / self.meaningful_len)
        # gray image
        if len(origin_img.shape) == 3:
            self.gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        elif len(origin_img.shape) == 2:
            self.gray_img = origin_img
        else:
            raise ValueError("shape of image can not be <=1 .")
        # gaussian filter
        if gauss_sigma > 0 and gauss_half_size > 0:
            gauss_size = 2 * gauss_half_size + 1
            self.filtered_img = cv2.GaussianBlur(self.gray_img, ksize=(gauss_size, gauss_size), gauss_sigma)
        else:
            raise ValueError("gauss_sima={} and gauss_half_size={} are forbidden".format(gauss_sigma, gauss_half_size))
        # gradient map
        self.dx = 

    def mtline_detect(self, origin_img, gauss_sigma, gauss_half_size):
        """
        Input: 
                 
            origin_img: numpy 2D array, gray scale
            gauss_sigma: sigma for gaussian smoothing
            gauss_half_size: kernel size
        Output:
            lines: [[f1,f2],[],[],...,[]]
               
        """
