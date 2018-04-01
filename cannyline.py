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
        
        # self.meaningful_len
        # self.threshold_grad_low
        # self.threshold_grad_high
        # self.canny_edge

        # self.num_row
        # self.num_col
        # self.threshold_search_steps

        # self.n4
        # self.n2

        # self.filtered_img
        # self.grad_map
        # self.orient_map
        # self.orient_map_int
        # self.search_map
        # self.mask

        # self.grad_points
        # self.grad_values
        # self.greater_than
        # self.smaller_than
    
    def getInfo(self, origin_img, gauss_sigma, gauss_half_size, p):
        """
        update paramter and bind them with self
        """
        gray_level_num = 255
        aperture_size = 3
        angle_per = np.pi / 8.0
        gauss_noise = 1.3333
        self.threshold_grad_low = gauss_noise
        
        if len(origin_img.shape) == 2:
            self.num_row, self.num_col = origin_img.shape
        else:
            self.mum_row, self.num_col, *_ = origin_img.shape

        self.n4 = np.square(self.num_row * self.num_row)

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
            self.filtered_img = cv2.GaussianBlur(self.gray_img, (gauss_size, gauss_size), gauss_sigma)
        else:
            raise ValueError("gauss_sima={} and gauss_half_size={} are forbidden".format(gauss_sigma, gauss_half_size))
        # gradient map
        self.dx = cv2.Sobel(self.filtered_img, cv2.CV_16S,1,0,ksize=aperture_size,scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
        self.dy = cv2.Sobel(self.filtered_img, cv2.CV_16S,0,1,ksize=aperture_size,scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)

        self.grad_map = np.abs(self.dx) + np.abs(self.dy)
        self.orient_map = np.arctan2(self.dx, -self.dy)
        self.orient_map_int = (self.orient_map + np.pi) / angle_per
        self.orient_map_int[np.abs(self.orient_map_int - 16) < 1e-8] = 0
        self.orient_map_int = self.orient_map_int.astype(np.uint8)
        
        # construct histogram
        histogram = np.zeros(shape=(8*gray_level_num), dtype=np.int32)
        total_num = 0
        for r_idx in range(self.num_row):
            for c_idx in range(self.num_col):
                grad = self.grad_map[r_idx, c_idx]
                if grad > self.threshold_grad_low:
                    histogram[int(grad+0.5)] += 1
                    total_num += 1
                else:
                    self.grad_map[r_idx,c_idx] = 0
        # gradient statistic
        self.n2 = np.sum(histogram * (histogram - 1))

        p_max = 1.0 / np.exp(np.log(self.n2) / self.meaningful_len)
        p_min = 1.0 / np.exp(np.log(self.n2) / np.sqrt(self.num_col*self.num_row))

        self.greater_than = np.zeros((8*gray_level_num,), dtype=np.float32)
        self.smaller_than = np.zeros((8*gray_level_num,), dtype=np.float32)

        self.greater_than = np.cumsum(histogram[::-1])[::-1] / total_num
        for i in range(8*gray_level_num-1, -1, -1):
            if(self.greater_than[i] > p_max):
                self.threshold_grad_high = i
                break
        for i in range(8*gray_level_num-1, -1, -1):
            if(self.greater_than[i] > p_min):
                self.threshold_grad_low = i
                break
        if(self.threshold_grad_low < gauss_noise):
            self.threshold_grad_low = gauss_noise
        # convert probabilistic meaningful to visual meaningful
        self.threshold_grad_high = np.sqrt(self.threshold_grad_high * self.visual_meaning_grad)

        # compute canny edge
        self.canny_edge = cv2.Canny(self.filtered_img, self.threshold_grad_low, self.threshold_grad_high, aperture_size) 

        # construct mask
        grad_rows, grad_cols = np.where(self.canny_edge > 0)
        self.mask = (self.canny_edge > 0).astype(np.uint8)

        self.grad_points = [(c,r)for r,c in zip(grad_rows, grad_cols)]
        self.grad_values = self.grad_map[grad_rows, grad_cols]
        # end get information, this part can also be called CannyPF



    def smart_routing(self, min_deviation, min_size):
        """
        return a list of cluster
        """
        if min_size < 3:
            min_size = 3
        
        mask_img_origin = np.copy(self.mask)
        ori_map_int = self.orient_map_int

        # sort descent
        descent_idx = np.argsort(-self.grad_values) # sort descent
        self.grad_values = self.grad_values[descent_idx]
        self.grad_points = [self.grad_points[i] for i in descent_idx]
        # find all pixels in meaningful line
        mask = mask_img_origin # just change its value
        orient_map = ori_map_int # values in this matrix will not be changed
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
        

        edge_chain = [] # [[(x1,y1), (x2,y2),...],[(x1,y1),(x2,y2)],...]
        # mask is used to reduce infinity loop

        for i in range(len(self.grad_points)):
            x = self.grad_points[i][0] # col
            y = self.grad_points[i][1] # row
            chain = []

            while True:
                chain.append((x,y))
                mask[y,x] = 0
                res, point = has_next(x,y)
                newx, newy = point
                if not res:
                    break
                x = newx
                y = newy
            # find pixels at the beginning of the edge
            x = self.grad_points[i][0] # col
            y = self.grad_points[i][1] # row
            res, point = has_next(x,y)
            if res:
                while True:
                    chain.append(point)
                    mask[point[1], point[0]] = 0
                    newres, point = has_next(*point)
                    if not newres:
                        break
            if len(chain) > self.meaningful_len:
                edge_chain.append(chain)
       
        # find segments. segments = [ [(col1,row1), (col2,row2), ..], [(col1,row1),..],..] 
        # find segment for each edge in edge_map
        print("length of strings = {}".format(len(edge_chain)))
        segments = []
        for i in range(len(edge_chain)):
            self.sub_division(segments, edge_chain[i],0, len(edge_chain[i])-1, min_deviation, min_size)
        
        return segments

    
    def sub_division(self,segments, edge, first_idx, last_idx, min_deviation, min_size):
        """
        input:
            
            edge, [(col1,row1), (col2,row2), ..]
            min_deviation:
            min_size:
        
        output: segment, it is chanaged inside the function
        """
        first_x = edge[first_idx][0] # col
        first_y = edge[first_idx][1] # row

        last_x = edge[last_idx][0] # col
        last_y = edge[last_idx][0] # row

        length = np.sqrt(
            np.square(first_x-last_x) + np.square(last_y - first_y))
        
        # find maximum deviation from line segment
        coord = np.array(edge, dtype=np.float32)
        coord -= np.array([first_x, first_y])
        dev = np.abs(coord[:,0] * (first_y - last_y) + coord[:,1] * (last_x - first_x))
        # note the deviation calculation
        max_dev_index = np.argmax(dev)
        max_dev = dev[max_dev_index]

        max_dev /= length

        # compute the ratio between the length of the segment an the max deviation
        # test the number of pixels of the sub clusters

        half_min_size = min_size / 2
        if all([
            max_dev >= min_deviation,
            max_dev_index - first_idx + 1 >= half_min_size,
            last_idx - max_dev_index + 1 >= half_min_size]):

            self.sub_division(segments, edge, first_idx, max_dev_index, min_deviation, min_size)
            self.sub_division(segments, edge, max_dev_index, last_idx, min_deviation, min_size)
        else:
            segments.append([edge[i] for i in range(first_idx, last_idx+1)])

    def least_square_fit(self, edge, sigma):
        """
        least square fitting, edge: [(c1, r1), (c2,r2), ...]
        return 4 parameters and updated edge
        """
        if edge[0][0] == edge[-1][0]:
            slope = float('inf')
        else:
            slope = (edge[-1][1]-edge[0][1]) / (edge[-1][0] - edge[0][0])
        coord = np.array(edge, dtype=np.float32)
        if np.abs(slope) < 1:
            sumxy = np.sum(coord, axis=0)
            sum_x = sumxy[0]
            sum_y = sumxy[1]
            sum_x2 = np.sum(np.square(coord[:,0]))
            sum_xy = np.sum(coord[:,0] * coord[:,1])
            n = len(edge)
            b = (sum_x2 * sum_y - sum_x * sum_xy) / (n*sum_x2-sum_x*sum_x)
            k = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x*sum_x)
            offsets = coord[:,1] - k*coord[:,0] - b
            dev = np.sum(np.square(offsets))
           
            # compute start and end
            start = 0
            end = n-1
            idx = 0
            dev_outliers = 0
            for i in range(n):
                if offsets[i] < 1.0:
                    idx += 1
                if idx == 2:
                    start = i
                    break
                else:
                    dev_outliers += np.square(offsets[i])
                
            idx = 0
            for i in range(n-1, -1, -1):
                if offsets[i] < 1.0:
                    idx += 1
                if idx == 2:
                    end = i
                    break
                else:
                    dev_outliers += np.square(offsets[i])
            if end <= start:
                return [], (None, None, None, None)
            # print("n={}".format(n))
            dev = np.sqrt(np.abs(dev-dev_outliers) / (n-2) )
            # print(dev)
            updated_edge = [edge[i] for i in range(start, end+1)]
            return updated_edge, (0, k, b, dev)
        else:
            sumxy = np.sum(coord, axis=0)
            sum_x = sumxy[0]
            sum_y = sumxy[1]
            sum_y2 = np.sum(np.square(coord[:,1]))
            sum_xy = np.sum(coord[:,0] * coord[:,1])
            n = len(edge)
            b = (sum_y2 * sum_x - sum_y * sum_xy)/ (n*sum_y2 - sum_y*sum_y)
            k = (n*sum_xy - sum_x * sum_y) / (n*sum_y2 - sum_y*sum_y)
            offsets = coord[:,1] - k*coord[:,0] - b
            dev = np.sum(np.square(offsets))

            start = 0
            end = n-1
            idx = 0
            dev_outliers = 0
            for i in range(n):
                if offsets[i] < 1.0:
                    idx += 1
                if idx == i:
                    start = i
                    break
                else:
                    dev_outliers += np.square(offsets[i])
            idx = 0
            for i in range(n-1, -1, -1):
                if offsets[i] < 1.0:
                    idx += 1
                if idx == 2:
                    end = i
                    break
                else:
                    dev_outliers += np.square(offsets[i])
            if end <= start: 
                return [], (None, None, None, None)
            
            dev = np.sqrt(np.abs(dev-dev_outliers) / (n-2))
            # print("n={}".format(n))

            updated_edge = [edge[i] for i in range(start, end+1)]
            return updated_edge, (1, k, b, dev)


    def get_metalines(self, segments, sigma):
        """
        return meta lines
         metalines[(id_num, direction, k, b,(start_x, start_y), (end_x, end_y)), (...),...]
         newsegments, locations in newsegments is corresponfing to metalines with same index
        upate mask, update segments,
        """
       
        newsegments = []
        lines = []
        count = 0
        for edge in segments:
            res_edge, para = self.least_square_fit(edge, sigma)
            if res_edge:
                count += 1
                newsegments.append(res_edge)
                for x, y in res_edge:
                    self.mask[y,x] = -count
                id_num = count
                direction, k, b, _ = para
                # direction :0 for k<1, 1 for k >=1
                if direction == 0:
                    start_x = res_edge[0][0]
                    start_y = k * start_x + b 
                    end_x = res_edge[-1][0]
                    end_y = k * end_x + b
                else: # direction == 1
                    start_y = res_edge[0][1]
                    start_x = k * start_y+ b
                    end_y = res_edge[-1][1]
                    end_x = k * end_y + b
                
                lines.append((id_num, direction, k, b, start_x, start_y, end_x, end_y))
        assert(len(newsegments) == len(lines))
        return newsegments, lines




    def mtline_detect(self, origin_img, gauss_sigma, gauss_half_size):
        """
        Input: 
                 
            origin_img: numpy 2D array, gray scale
            gauss_sigma: sigma for gaussian smoothing
            gauss_half_size: kernel size
        Output:
            lines: [[f1,f2],[],[],...,[]]
               
        """
        self.getInfo(origin_img, gauss_sigma, gauss_half_size, self.p)

        # smart routing
        min_deviation = 2.0
        min_size = self.meaningful_len / 2
        segments = self.smart_routing(min_deviation, min_size)
        
        # TODO the number of segments is different but close
        print("length of segments = {}".format(len(segments)))
        print("done")

        # get initial metaline
        
        segments, metalines = self.get_metalines(segments, self.sigma)

        print("length of segments = {}".format(len(segments)))
        print("length of metalines = {}".format(len(metalines)))

        # TODO

        # test
        # result_img = np.zeros((origin_img.shape),dtype=np.uint8)
        # for chain in edge_chain:
        #     for col,row in chain:
        #         result_img[row, col] = 255
        # print("done")
        # print('length edge_chain = {}'.format(len(edge_chain)))
        # cv2.imwrite("./img/testedgemap.jpg", result_img)


