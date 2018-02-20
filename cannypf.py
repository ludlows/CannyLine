
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



    def comp_edge_map(self):
        """
        a wrapper for canny detector in OpenCV
        :return: numpy array
        """
        low, high = self.comp_threshold()
        return cv2.Canny(self.img, low, high)
