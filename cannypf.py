
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
        print("meaningful_length", meaningful_length)
        angle = 2 * np.arctan(2 / meaningful_length)
        smth_img = cv2.GaussianBlur(self.gray_img, (self.gauss_size, self.gauss_size), 1.0)
        self.smth_img = smth_img
        # compute gradient map and orientation map
        gradient_map = np.zeros((num_row, num_col))
        dx = cv2.Sobel(smth_img,cv2.CV_64F,1,0,ksize=3,scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
        dy = cv2.Sobel(smth_img,cv2.CV_64F,0,1,ksize=3,scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)

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
        print(" N2 = ", num_p)
        #
        p_max = 1.0 / np.exp(np.log(num_p)/meaningful_length)
        p_min = 1.0 / np.exp(np.log(num_p)/np.sqrt(num_row * num_col))
        print('p_max', p_max)
        print('p_min', p_min)
        print("hist[8*graylevels-1]= ", hist[gray_levels*8-1])
        count = 0
        prob = np.zeros((8*gray_levels))
        for i in range(8*gray_levels-1, -1, -1):
            count += hist[i]
            prob[i] = count / total_num
      
        print("prob[8*255-1] = ", prob[8*255-1])
        # compute two threshold
        high_threshold = 0
        low_threshold = 1.3333
        
        for i in range(8*gray_levels-1, -1,-1):
            p_cur = prob[i]
            if p_cur > p_max:
                high_threshold = i
                break
        for i in range(8*gray_levels-1, -1,-1):
            p_cur = prob[i]
            if p_cur > p_min:
                low_threshold = i
                break
        if low_threshold < 1.3333:
            low_threshold = 1.3333
        # visual meaningful high threshold
        high_threshold = np.sqrt(high_threshold * self.vm_grad)
        print('low_threshold     high_threshold')
        print(low_threshold, high_threshold)
        return low_threshold, high_threshold
        
    def comp_edge_map(self):
        """
        a wrapper for canny detector in OpenCV
        :return: numpy array
        """
        low, high = self.comp_threshold()
        return cv2.Canny(self.smth_img, low, high, apertureSize=3)