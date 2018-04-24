
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


def comp_edge_chain(image, edge_map):
    """
    this function compute list of line, based on edge map
    ------
    Input: image, image, numpy array.
           edge_map, 2d numpy array, computed by CannyPF
    Output: list of points
    """
    dim_len = len(image.shape)
   
    if dim_len == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif dim_len == 2:
        pass
    else:
        raise ValueError("number of channels in image is not right")
    
    num_row, num_col = image.shape

    # compute gradient
    dx = cv2.Sobel(image, cv2.CV_16S,1,0,ksize=3,scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(image, cv2.CV_16S,0,1,ksize=3,scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)

    angleper = np.pi / 8.0
    grad_map = np.abs(dx) + np.abs(dy)
    
    orient_map = (np.arctan2(dx, -dy) + np.pi) / angleper 
    orient_map[np.abs(orient_map - 16) < 1e-8] = 0 # 2pi case
    orient_map = orient_map.astype(np.uint8)
    # compute edge chains
    edge_map = edge_map.astype(np.uint8)
    rows, cols = np.where(edge_map > 1e-8)
    # col, row
    gradient_points = [(c,r) for r,c in zip(rows, cols)]
    gradient_values = grad_map[(rows, cols)]
    
    
   
    mask = (edge_map > 1e-8).astype(np.uint8)

    order = np.argsort(gradient_values)
    
    gradient_points = [gradient_points[i] for i in reversed(order)]
    gradient_values = [gradient_values[i] for i in reversed(order)]   

    def has_next(x_seed, y_seed):
        """
        this function return boolean result.
        Check whether there is a next value
        Input: x_seed, int, col
               y_seed, int, row
        Output: (boolean, (col, row)
        """
        num_row, num_col = mask.shape
        direction = orient_map[y_seed, x_seed]
        direction0 = direction - 1
        if direction0 < 0:
            direction0 = 15
        direction1 = direction
        direction2 = direction + 1
        if np.abs(direction2 -16) < 1e-8:
            direction2 = 0
        
        x_offset = [0, 1, 0, -1, 1, -1, -1, 1]
        y_offset = [1, 0, -1, 0, 1, 1, -1, -1]
        directions = np.array([direction0, direction1, direction2], dtype=np.float32)
        for i in range(8):
            x = x_seed + x_offset[i]
            y = y_seed + y_offset[i]
            if (x >= 0 and x < num_col) and (y >= 0 and y < num_row):
                if mask[y, x] > 0 :
                    temp_direction = orient_map[y, x]
                    if any(np.abs(directions - temp_direction) < 1e-8 ):
                        return (True, (x, y)) # (boolean, (col, row)) 
        return (False, (None, None))
    # to find strings 
    meaningful_length = int(2.0 * np.log(num_row*num_col)/ np.log(8.0) + 0.5)
    # edge_chain , the [[(c,r),..... ,(c,r)], [(c,r),...(c,r)],...] need to be returned
    edge_chain = []
    print("start computing edge chain")
    print("num of gradient points: {}".format(len(gradient_values)))
    maximal_length = int(np.sqrt(num_col**2+num_row**2))
    # mask is used to reduce infinity loop
    for i in range(len(gradient_points)):
        # print("i = {}".format(i))
        x = gradient_points[i][0] # col
        y = gradient_points[i][1] # row
        chain = []
        
        while True:
            # print("i = {}, col = {}, row = {}".format(i, x,y))
            chain.append((x,y))
            mask[y, x] = 0
            res, point = has_next(x,y)
            newx, newy = point
            if not res:
                break
            if len(chain) >= maximal_length:
                break
            else:
                x = newx
                y = newy
        # find pixels at the begining of the string
        x = gradient_points[i][0] # col
        y = gradient_points[i][1] # row
        res, point = has_next(x,y)
        if res:
            while True:
                chain.append(point)
                mask[point[1], point[0]] = 0
                newres, point = has_next(*point)
                if not newres:
                    break
        
        if (len(chain) > meaningful_length):
            edge_chain.append(chain)
    print("end")
    return edge_chain


def color_imwrite(edge_chain, shape, name='out.jpg'):
    colors = [(int(np.random.random()*255),
               int(np.random.random()*255),
               int(np.random.random()*255)) for _ in range(29)]
    img = 255 * np.ones(shape, dtype=np.uint8)
    for idx, chain in enumerate(edge_chain):
        for x, y in chain:
            img[y, x, :] = colors[ idx % 29]
    cv2.imwrite(name, img)


        

     