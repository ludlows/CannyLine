
import cv2
import numpy as np


class CannyPF(object):
    """
    pass
    """
    def __init__(self, gauss_size, vm_grad, img):
        """
        initialize parameters and applying gaussian smooth filter to original image
        ------------------------------------------------
        :param gauss_size: int, gaussian smooth filter size
        :param vm_grad: float
        :param img: 2D or 3D numpy array represents image gary matrix
        """
        self.gauss_size = gauss_size
        self.vm_grad = vm_grad
        if len(img.shape) > 2:
            self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            self.gray_img = img
        else:
            raise ValueError("shape of image can not be <=1 .")


    def comp_threshold(self):
        """
        this function computes thresholds, which will be used in comp_edge_map
        :return: (low_thresh, high_thresh)
        """
        num_row, num_col = self.gray_img.shape
        # compute meaningful length
        meaningful_length = int(2.0 * np.log(num_row * num_col) / np.log(8) + 0.5)
        angle = 2 * np.arctan(2 / meaningful_length)
        smth_img = cv2.GaussianBlur(self.gray_img, self.gauss_size, 1.0)
        # compute gradient map and orientation map
        gradient_map = np.zeros((num_row, num_col))
        dx = cv2.Sobel(smth_img,cv2.CV_64F,1,0,ksize=3)
        dy = cv2.Sobel(smth_img,cv2.CV_64F,0,1,ksize=3)

        # construct a histogram
        gray_levels = 255
        total_num = 0
        grad_low = 1.3333
        hist = np.zeros((8*gray_levels,))
        for ind_r in range(num_row):
            for ind_c in range(num_col):
                grd = np.abs(dx[ind_r, ind_c]) + np.abs(dy[ind_r, ind_c])
                if grd > grad_low:
                    hist[int(grd + 0.5)] += 1
                    total_num += 1
                    gradient_map[ind_r, ind_c] = grd
                else:
                    gradient_map[ind_r, ind_c] = 0
        # gradient statistic
        num_p = np.sum(hist * (hist-1))
        #
        p_max = 1.0 / np.exp(np.log(num_p)/meaningful_length)
        p_min = 1.0 / np.exp(np.log(num_p)/np.sqrt(num_row * num_col))
        prob = np.cumsum(hist) / total_num
        # compute two threshold
        high_threshold = 0
        low_threshold = 1.3333
        hist_length = hist.shape[0]

        for i in range(hist_length):
            p_cur = prob[-i-1]
            if p_cur > p_max:
                high_threshold = hist_length - i - 1
                break
        for i in range(hist_length):
            p_cur = prob[-i-1]
            if p_cur > p_min:
                low_threshold = hist_length - i - 1
                break
        if low_threshold < 1.3333:
            low_threshold = 1.3333
        # visual meaningful high threshold
        high_threshold = np.sqrt(high_threshold * self.vm_grad)
        return cv2.Canny(smth_img, low_threshold, high_threshold, apertureSize=3)







    def comp_edge_map(self):
        """
        a wrapper for canny detector in OpenCV
        :return: numpy array
        """
        low, high = self.comp_threshold()
        return cv2.Canny(self.img, low, high)
