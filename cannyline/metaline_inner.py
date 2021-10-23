"""
this repository implements canny line
author: github.com/ludlows
2018-04-10
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
        self.sigma = 4.0
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
            self.num_row, self.num_col, *_ = origin_img.shape

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
        histogram = np.zeros(shape=(8*gray_level_num), dtype=np.int64)
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
        self.n2 = np.sum(histogram * np.abs(histogram - 1))
        print(" self. n2", self.n2)

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
        if not hasattr(self, 'threshold_grad_high'):
            self.threshold_grad_high = 15 * self.threshold_grad_low

        self.threshold_grad_high = np.sqrt(self.threshold_grad_high * self.visual_meaning_grad)

        # compute canny edge
        self.canny_edge = cv2.Canny(self.filtered_img, self.threshold_grad_low, self.threshold_grad_high, aperture_size) 

        # construct mask
        grad_rows, grad_cols = np.where(self.canny_edge > 0)
        self.mask = (self.canny_edge > 0).astype(np.uint8)
        self.mask = self.mask.astype(np.int32)

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
                # chain.sort()
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
            dev = np.sqrt(np.abs(dev-dev_outliers) / (n-2) )
            if end <= start:
                return [], (0, k, b, dev)
            # print("n={}".format(n))
            
            # print(dev)
            # print("length of edge", len(edge))
            # print("n  is ", n)
            # print('end is ', n)
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
           
           
            dev = np.sqrt(np.abs(dev-dev_outliers) / (n-2))
            # print("n={}".format(n))
            if end <= start: 
                return [], (1, k, b, dev)
            

            updated_edge = [edge[i] for i in range(start, end+1)]
            return updated_edge, (1, k, b, dev)


    def get_metalines(self, segments, sigma):
        """
        return meta lines
         metalines[(id_num, direction, k, b, start_x, start_y, end_x, end_y), (...),...]
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
                    # print("-count", -count)
                    # print("mask")
                    # print(self.mask[y,x])
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
    
    def merge_lines(self, id_num, line_idx, line_hyp, threshold_angle):
        """
        id_num, in self.metaline[(id_num, direction, k, b, start_x, start_y, end_x, end_y), (...),...]
        line_hyp: a list of index of metalines
        threshold_angle: float
        line_idx: can be used to access current segment by self.segments[line_idx], 
                                        current metaline by self.metalines[line_idx]
        """
        new_id_num, _, _, _, start_x, start_y, end_x, end_y = self.metalines[line_idx]
        assert(new_id_num == id_num)
        start_x = self.segments[line_idx][0][0]
        start_y = self.segments[line_idx][0][1]
        end_x = self.segments[line_idx][-1][0]
        end_y = self.segments[line_idx][-1][1]
        
        if start_x == end_x:
            current_angle = np.pi / 2
        else:
            current_angle = np.arctan((start_y - end_y) / (start_x - end_x)) 
        
        # metaline merge judge
        
        angles = []
        for v in line_hyp:
            new_id_num, _, _, _, start_x, start_y, end_x, end_y = self.metalines[v]
            start_x = self.segments[v][0][0]
            start_y = self.segments[v][0][1]
            end_x = self.segments[v][-1][0]
            end_y = self.segments[v][-1][1]
            if start_x == end_x:
                angles.append(np.pi/2)
            else:
                angles.append(np.arctan((start_y - end_y) / (start_x - end_x)) )
        
        # set angle min
        # get line index with min angle
        angle_min = 100
        min_line_idx = 0
        for angle, idx in zip(angles, line_hyp):
            offset = min(np.abs(current_angle - angle), np.pi - np.abs(current_angle - angle))
            if offset < angle_min:
                angle_min = offset
                min_line_idx = idx
        # failed 
        if angle_min > threshold_angle:
            return -1 
        # merge
        threshold_dist = 4
        _, _, _, _, cur_start_x, cur_start_y, cur_end_x, cur_end_y = self.metalines[line_idx] # input index
        cur_start_x = self.segments[line_idx][0][0]
        cur_start_y = self.segments[line_idx][0][1]
        cur_end_x = self.segments[line_idx][-1][0]
        cur_end_y = self.segments[line_idx][-1][1]
        k = np.abs(np.tan(current_angle))
        if k > 1:
            _, _, _, _, start_x, start_y, end_x, end_y = self.metalines[min_line_idx]
            start_x = self.segments[min_line_idx][0][0]
            start_y = self.segments[min_line_idx][0][1]
            end_x = self.segments[min_line_idx][-1][0]
            end_y = self.segments[min_line_idx][-1][1]
            dist_start = np.abs(start_y - cur_end_y)
            dist_end = np.abs(end_y - cur_end_y)
            if dist_start < dist_end and dist_start < threshold_dist:
                if (end_y - cur_end_y) * (cur_start_y - cur_end_y) < 0:
                    # start merging
                    self.segments[line_idx].extend(self.segments[min_line_idx])
                    return min_line_idx # return removed index
            if dist_end <dist_start and dist_end < threshold_dist:
                if (start_y - cur_end_y) * (cur_start_y - cur_end_y) < 0:
                    # start merging reversed version
                    self.segments[line_idx].extend(self.segments[min_line_idx][::-1])
                    return min_line_idx # return removed index
            return -1
        else: # 0<=k <=1
            _, _, _, _, start_x, start_y, end_x, end_y = self.metalines[min_line_idx]
            start_x = self.segments[min_line_idx][0][0]
            start_y = self.segments[min_line_idx][0][1]
            end_x = self.segments[min_line_idx][-1][0]
            end_y = self.segments[min_line_idx][-1][1]
            dist_start = np.abs(start_x - cur_end_x)
            dist_end = np.abs(end_x - cur_end_x)
            if dist_start < dist_end and dist_start < threshold_dist:
                if (end_x - cur_end_x) * (cur_start_x - cur_end_x) < 0:
                    # start merging
                    self.segments[line_idx].extend(self.segments[min_line_idx])
                    return min_line_idx
            if dist_end < dist_start and dist_end < threshold_dist:
                if (start_x - cur_end_x) * (cur_start_x - cur_end_x) < 0:
                    # start merging reversed version
                    self.segments[line_idx].extend(self.segments[min_line_idx][::-1])
                    return min_line_idx
            return -1
        

    def extend_hori_line(self, cur_line_idx, removal):
        """
        cur_line_idx: one index of self.metalines and self.segments
        extend horizontal line
        remove: sign for whether a sements and metalines should be ignored
        """
        id_num, _, meta_k, meta_b, start_x, _, end_x, end_y = self.metalines[cur_line_idx]
        start_x = self.segments[cur_line_idx][0][0]
        # start_y = self.segments[cur_line_idx][0][1]
        end_x = self.segments[cur_line_idx][-1][0]
        end_y = self.segments[cur_line_idx][-1][1]
        assert(cur_line_idx == id_num - 1)
        end_y = meta_k * end_x + meta_b
        cur_x = end_x
        cur_y = end_y
        x_initial = int(cur_x + 0.5)
        y_initial = int(cur_y + 0.5)
        if end_x == start_x:
            index = 0
        else:
            index = (end_x - start_x) / np.abs(end_x - start_x)

        # the change of former_points should be upadated into self.segments
        former_segment = [v for v in self.segments[cur_line_idx]]
        edge = 0
        edge_total = 0
        extend = False
        gap = 0

        while True:
            # print("extend horizontal line ", id_num)
            x_initial += index
            cur_y += index*meta_k
            y_initial = int(cur_y + 0.5)

            choose_up = False
            if y_initial + 0.5 > cur_y:
                choose_up = True
            if x_initial > 0 and x_initial < self.num_col -1 and y_initial > 0 and y_initial < self.num_row -1:
                m0 = self.mask[int(y_initial), int(x_initial)]
                m1 = self.mask[int(y_initial)-1, int(x_initial)]
                m2 = self.mask[int(y_initial)+1, int(x_initial)]
                hypo_line_id_nums = []
                if m0 < 0 and m0 != -id_num:
                    hypo_line_id_nums.append(-m0-1)
                if m1 < 0 and m1 != -id_num:
                    hypo_line_id_nums.append(-m1-1)
                if m2 < 0 and m2 != -id_num:
                    hypo_line_id_nums.append(-m2-1)
                if hypo_line_id_nums:
                    remove_index = self.merge_lines(id_num, id_num-1, hypo_line_id_nums, self.thresh_angle)
                    # after this step
                    # self.segments[cur_line_index] changed
                    if remove_index != -1: # merging happents
                        new_segment, para = self.least_square_fit(self.segments[cur_line_idx], self.sigma)
                        if new_segment:
                            # update self.segments
                       
                            self.segments[cur_line_idx] = new_segment
                            # # update meta lines
                            # temp_meta_value = list(self.metalines[cur_line_idx])
                            # temp_meta_value[-4:] = [new_segment[0][0], new_segment[0][1], new_segment[-1][0], new_segment[-1][1]]
                            # self.metalines[cur_line_idx] = tuple(temp_meta_value)
                            # # important above
                            direction, meta_k, meta_b, _ = para
                            if direction == 0:
                                x_initial = new_segment[-1][0]
                                cur_y = meta_k*x_initial + meta_b
                                removal[remove_index] = 1
                                extend = True
                            elif direction == 1:
                                k = meta_k
                                b = meta_b
                                new_y_start = new_segment[0][1]
                                new_x_start = k * new_y_start + b
                                new_y_end = new_segment[-1][1]
                                new_x_end = k * new_y_end + b
                                self.metalines[cur_line_idx] = (cur_line_idx+1, direction, k, b, new_x_start, new_y_start, new_x_end, new_y_end)
                                # TODO
                                self.extend_verti_line(cur_line_idx, removal)
                                extend = True
                            else:
                                raise ValueError("the direction value is invalid here")
                            
                            # update mask
                            for x,y in self.segments[cur_line_idx]:
                                self.mask[y,x] = -id_num
                        else:
                            break
                    else:
                        break
                else:
                    if any([m0==1, m1==1, m2==1]) and m0+m1+m2 == 1:
                        if m0 == 1 and m0 >= m1 and m0 >= m2:
                            self.segments[cur_line_idx].append((int(x_initial), int(y_initial)))
                            self.mask[int(y_initial), int(x_initial)] = -id_num
                        elif m1 == 1 and m1>= m0  and m1>=m2:
                            self.segments[cur_line_idx].append((int(x_initial), int(y_initial) - 1))
                            self.mask[int(y_initial-1), int(x_initial)] = -id_num
                        elif m2 == 1 and m2 >= m0  and m2 >= m1:
                            if choose_up:
                                gap += 1
                                continue
                            self.segments[cur_line_idx].append((int(x_initial), int(y_initial+1)))
                            self.mask[int(y_initial)+1, int(x_initial)] = -id_num
                        edge += 1
                        edge_total += 1
                    else:
                        gap += 1
                    if edge==0 or gap / edge >= 0.25:
                        break
                    if gap == 2:
                        edge = 0
                        gap = 0
                    if edge_total >= self.meaningful_len:
                        new_segment, para = self.least_square_fit(self.segments[cur_line_idx], self.sigma)
                        # this least_square_fit is for computing Parameters only
                        # if new_segment:
                        #     # update self.segments
                        #     self.segments[cur_line_idx] = new_segment
                        #     self.metalines[cur_line_idx][-4:] = 
                        meta_k = para[1]
                        meta_b = para[2]
                        cur_y = x_initial * meta_k + meta_b
                        edge_total = 0
                        gap = 0
                        extend = True
                        # else:
                        #     raise Exception("least square fitting failed")
            else:
                break 
        if extend:
            _, para = self.least_square_fit(self.segments[cur_line_idx], self.sigma)
            direction, k, b, _ = para
            if direction == 0:
                new_x_start = self.segments[cur_line_idx][0][0]
                new_y_start = k * new_x_start + b
                new_x_end = self.segments[cur_line_idx][-1][0]
                new_y_end = new_x_end * k + b
            elif direction == 1:
                new_y_start = self.segments[cur_line_idx][0][1]
                new_x_start = k * new_y_start + b
                new_y_end = self.segments[cur_line_idx][-1][1]
                new_x_end = k * new_y_end + b
            
            self.metalines[cur_line_idx] = (cur_line_idx+1, direction, k, b, new_x_start, new_y_start, new_x_end, new_y_end)
        else:
            self.segments[cur_line_idx] = former_segment



    
    def extend_verti_line(self, cur_line_idx, removal):
        """
        extend vertical line
        """
        id_num, _, meta_k, meta_b, _, start_y, _, end_y = self.metalines[cur_line_idx]
        assert(cur_line_idx == id_num -1)
        start_y = self.segments[cur_line_idx][0][1]
        end_y = self.segments[cur_line_idx][-1][1]

        end_x = meta_k * end_y + meta_b
        cur_x = end_x
        cur_y = end_y

        x_initial = int(cur_x + 0.5)
        y_initial = int(cur_y + 0.5)

        if end_y == start_y:
            index = 0
        else: 
            index = (end_y - start_y) / np.abs(end_y - start_y)

        former_points = [v for v in self.segments[cur_line_idx]]

        gap = 0
        edge = 0
        edge_total = 0

        extend = False

        pre_initial = (None, None)

        while True:
            # print("extend vertical line ", id_num)
            # print("k is ", meta_k)
            y_initial += index
            # print(index)
            y_initial = int(y_initial)
            cur_x += index * meta_k
            x_initial = int(cur_x + 0.5)
            # print("x_initial", x_initial, "y_inital", y_initial)
            # print("pre_initial", pre_initial)
            # print("self.mask[y_initial, x_initial]", self.mask[y_initial, x_initial])
            # if self.mask[y_initial, x_initial] == -id_num:
            #     break
            # TODO there is a infinite loop , how to fix it?
            if pre_initial == (x_initial, y_initial):
                break
            # print("len()", len(self.segments[cur_line_idx]))
            pre_initial = (x_initial, y_initial)

            choose_left = False
            if x_initial + 0.5 > cur_x:
                choose_left = True
            if x_initial > 0 and x_initial < self.num_col - 1 and y_initial > 0 and y_initial < self.num_row -1:

                m0 = self.mask[int(y_initial), int(x_initial)]
                m1 = self.mask[int(y_initial), int(x_initial) - 1]
                m2 = self.mask[int(y_initial), int(x_initial) + 1]
                hype_line_ids = []
                if m0 < 0  and m0 != -id_num:
                    hype_line_ids.append(-m0-1)
                if m1 < 0 and m1 != -id_num:
                    hype_line_ids.append(-m1-1)
                if m2 < 0 and m2 != -id_num:
                    hype_line_ids.append(-m2-1)
                if hype_line_ids:
                    remove_index = self.merge_lines(id_num, id_num-1, hype_line_ids, self.thresh_angle)
                    if remove_index != -1:
                        new_segment, para = self.least_square_fit(self.segments[cur_line_idx], self.sigma)
                        # TODO this step always add one (332,21) into list new_segment
                        # print('new_segment', new_segment)
                        if new_segment:
                            
                            self.segments[cur_line_idx] = new_segment
                            direction, meta_k, meta_b, _ = para
                            if direction == 1:
                                y_initial = new_segment[-1][1]
                                cur_x = meta_k * y_initial + meta_b
                                removal[remove_index] = 1
                                extend = True
                            elif direction == 0: # this step
                                # //0 for k<1, 1 for k >=1
                                k = meta_k
                                b = meta_b
                                new_x_start = new_segment[0][0]
                                new_y_start = k*new_x_start + b
                                new_x_end = new_segment[-1][0]
                                new_y_end = new_x_end * k + b
                                self.metalines[cur_line_idx] = (id_num, direction, k, b, new_x_start, new_y_start, new_x_end, new_y_end)
                                self.extend_hori_line(cur_line_idx, removal)
                            # update mask
                            for x,y in new_segment:
                                self.mask[y,x] = -id_num
                        else:
                            self.segments[cur_line_idx] = former_points
                            break
                    else:
                        break
                else:
                    if any([m0==1, m1==1, m2==1]) and m0+m1+m2 == 1:
                        if m0==1 and m0>=m1 and m0>=m2:
                            self.segments[cur_line_idx].append((int(x_initial), int(y_initial)))
                            self.mask[int(y_initial), int(x_initial)] = -id_num
                        elif m1 == 1 and m1>=m0 and m1>=m2:
                            self.segments[cur_line_idx].append((int(x_initial)-1, int(y_initial)))
                            self.mask[int(y_initial), int(x_initial)-1] = -id_num
                        elif m2 == 1 and m2>=m0 and m2>=m1:
                            if choose_left:
                                gap += 1
                                continue
                            self.segments[cur_line_idx].append((int(x_initial)+1, int(y_initial)))
                            self.mask[int(y_initial), int(x_initial)+1] = -id_num
                        edge += 1
                        edge_total += 1
                    else:
                        gap += 1
                    if edge == 0 or gap / edge >= 0.25 :
                        break
                    if gap == 2:
                        edge = 0
                        gap = 0
                    if edge_total >= self.meaningful_len:
                        _, para = self.least_square_fit(self.segments[cur_line_idx], self.sigma)
                        _,meta_k,meta_b,_ = para
                        cur_x = y_initial * meta_k + meta_b
                        edge_total = 0
                        gap = 0
                        extend = True
                        # TODO former 
                        former_points = self.segments[cur_line_idx]
            else:
                break
        if extend:
            _, para = self.least_square_fit(self.segments[cur_line_idx], self.sigma)
            direction, k, b, _ = para
            if direction == 0:
                new_x_start = self.segments[cur_line_idx][0][0]
                new_y_start = k * new_x_start + b
                new_x_end = self.segments[cur_line_idx][-1][0]
                new_y_end = new_x_end * k + b
                self.metalines[cur_line_idx] = (id_num, direction, k, b, new_x_start, new_y_start, new_x_end, new_y_end)
            elif direction == 1:
                new_y_start = self.segments[cur_line_idx][0][1]
                new_x_start = k* new_y_start + b
                new_y_end = self.segments[cur_line_idx][-1][1]
                new_x_end = k * new_y_end + b
                self.metalines[cur_line_idx] = (id_num,direction,k,b, new_x_start,new_y_start, new_x_end, new_y_end)
        else:
            self.segments[cur_line_idx] = former_points
                            


    def grad_weight_LSF(self, points, sigma):
        """
        gradient weighted Least Square Fitting
        """
        n = len(points)
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        if start_x == end_x:
            slope = float('inf')
        else:
            slope = ((start_y - end_y) / (start_x - end_x))
        weight = np.array([self.grad_map[y,x] for x,y in points], dtype=np.float32)
        weight_sum = np.sum(weight)
        weight = weight / weight_sum
        
        coord = np.array(points, dtype=np.float32)

        sum_x = np.sum(coord[:,0] * weight)
        sum_y = np.sum(coord[:,1] * weight)
        sum_xy = np.sum(coord[:,0] * coord[:,1] * weight)

        if np.abs(slope) < 1:
            sum_x2 = np.sum(np.square(coord[:,0]) * weight)
            b = (sum_x2 * sum_y - sum_x * sum_xy) / (sum_x2 - sum_x * sum_x)
            k = (sum_xy - sum_x * sum_y ) / (sum_x2 - sum_x * sum_x)
            offset = coord[:,1] - k * coord[:,0] - b 
            dev = np.sqrt(np.sum(np.square(offset)) / (n-2))
            if dev < sigma:
                return True, (0, k,b,dev)
            else:
                return False, (0,k,b,dev)
        else:
            sum_y2 = np.sum(np.square(coord[:,1]) * weight)
            b = (sum_y2 * sum_x - sum_y * sum_xy) / (sum_y2 - sum_y * sum_y)
            k = (sum_xy - sum_x * sum_y) / (sum_y2 - sum_y * sum_y)
            offset = coord[:,0] - k * coord[:,1] - b
            dev = np.sqrt(np.sum(np.square(offset))/ (n-2))
            if dev < sigma:
                return True, (1, k,b,dev)
            else:
                return False, (1,k,b,dev)
            

       
        

    def extend_lines(self,removal):
        """
        lines: [(id_num, direction, k, b, start_x, start_y, end_x, end_y), (...),...]
        segments: [[(col1,row1)...], [(col1,row1),...],...]
        removal: [0,...]
        """
        long_line_idx = [(len(self.segments[i]), i) for i in range(len(self.segments)) if len(self.segments[i]) > 2 * self.meaningful_len]
        long_line_idx.sort(reverse=True)
        for _, idx in long_line_idx:
            if not removal[idx]:
                direction = self.metalines[idx][1]
                if direction == 0: #  horizontal line
                    self.extend_hori_line(idx, removal)
                    # reverse self.segments[idx]
                    self.segments[idx].reverse()
                    if self.metalines[idx][1] == 0: # direction ==0
                        self.extend_hori_line(idx, removal)
                    elif self.metalines[idx][1] == 1:
                        self.extend_verti_line(idx, removal)
                    else:
                        raise ValueError(" direction is wrong")
                elif  direction == 1: # vertical line
                    self.extend_verti_line(idx, removal)
                    # reverse
                    self.segments[idx].reverse()
                    if self.metalines[idx][1] == 0:
                        self.extend_hori_line(idx, removal)
                    elif self.metalines[idx][1] == 1:
                        self.extend_verti_line(idx, removal)
                    else:
                        raise ValueError("directoin is wrong")
                    
                else:
                    raise ValueError("the direction sign can not be {}".format(direction))
                # grdient weight least square fitting
                _, para = self.grad_weight_LSF(self.segments[idx], 0.5)
                direction, k, b, _ = para
                if direction == 1:
                    start_y = self.segments[idx][0][1]
                    start_x = k * start_y + b
                    end_y = self.segments[idx][-1][1]
                    end_x = k * end_y + b
                elif direction == 0:
                    start_x = self.segments[idx][0][0]
                    start_y = k * start_x + b 
                    end_x = self.segments[idx][-1][0]
                    end_y = k * end_x + b
                self.metalines[idx] = (idx+1, direction, k, b, start_x, start_y, end_x, end_y)


    def line_valid_check(self, removal):
        """
        line valid checking
        """
        for idx in range(len(self.segments)):
            if removal[idx] or len(self.segments[idx]) < 2*self.meaningful_len:
                removal[idx] = 1
            else:
                orient_prob = self.line_valid_check_grad_orient(idx)
                grad_prob = self.line_valid_check_gradient(idx)
                if orient_prob * self.n4 * grad_prob * self.n2 > 1:
                    removal[idx] = 1
        return


    
    def line_valid_check_gradient(self, meta_line_idx):
        num_points = len(self.segments[meta_line_idx])
        step = int(num_points / self.meaningful_len)
        if step == 0:
            step = 1
        gradient = []
        j = 0
        while j < num_points:
            x,y = self.segments[meta_line_idx][j]
            gradient.append(self.grad_map[y,x])
            j += step
        gradient.sort(reverse=True)
        index = int(gradient[-1] + 0.5)
        prob = np.power(self.greater_than[index], num_points)
        return prob
    
    def line_valid_check_grad_orient(self, meta_line_idx):
        """
        pass
        metaLines.xs === self.metalines[4]
        """
        angle_offset = np.pi / 8
        start_x, start_y, end_x, end_y = self.metalines[meta_line_idx][-4:]
     
        delta_x = start_x - end_x
        delta_y = start_y - end_y
        if delta_x == 0:
            angle_line = np.pi / 2
        else:
            angle_line = np.arctan(delta_y / delta_x)
       
        count1 = 0
        count2 = 0
        count3 = 0
        for x,y in self.segments[meta_line_idx]:
            angle = self.orient_map[y,x]
            if np.abs(angle - angle_line) < angle_offset:
                count1 += 1
            if np.pi - np.abs(angle - angle_line) < angle_offset:
                count2 += 1
            count3 += 1
        count = max(count1,count2)
        return self.probability(count3, count, self.p)

    def probability(self, total_num, num, p):
        """
        """
        v = np.power(p, total_num)
        prob = v
        for i in range(total_num - num):
            v = v * (total_num - i) / (1+i) * (1-p) / p
            prob += v
        return prob





    def mtline_detect(self, origin_img, gauss_sigma, gauss_half_size):
        """
        Input: 
                 
            origin_img: numpy 2D array, gray scale
            gauss_sigma: sigma for gaussian smoothing
            gauss_half_size: kernel size
        Output:
            lines: [[start_x,start_y, end_x, end_y, id_num],[],[],...,[]]
               
        """
       
        self.getInfo(origin_img, gauss_sigma, gauss_half_size, self.p)
        print("meaningful length", self.meaningful_len)
        # smart routing
        min_deviation = 2.0
        min_size = self.meaningful_len / 2
        segments = self.smart_routing(min_deviation, min_size)
        
        #  the number of segments is different but close
        print("length of segments = {}".format(len(segments)))
        print("done")

        # get initial metaline
        
        segments, metalines = self.get_metalines(segments, self.sigma)

        print("length of segments = {}".format(len(segments)))
        print("length of metalines = {}".format(len(metalines)))
        print("min of self.mask = {}, should be a negative".format(np.min(self.mask)))
        # //meta line extending
        remove = [0 for _ in range(len(segments))] # sign
        # the order of metalines and segmenst will not change forever
        self.segments = segments
        self.metalines = metalines 
        assert(len(segments) == len(metalines))

        self.extend_lines(remove)
        print("length of self.segments", len(self.segments))
        print("length of meta lines", len(self.metalines))
        print('sum of remove', np.sum(remove))

        self.line_valid_check(remove)

        result_lines = []
        for i in range(len(remove)):
            if not remove[i]:
                points = self.segments[i]
                id_num, direction,_,_,start_x, start_y, end_x, end_y = self.metalines[i]
                _, para = self.grad_weight_LSF(points, self.sigma)
                direction, k, b, _ = para
                if direction == 0:
                    points.sort()
                    start_x = points[0][0]
                    end_x = points[-1][0]
                    if np.isinf(k):
                        start_y = points[0][1]
                        end_y = points[-1][1]
                    else:
                        start_y = k * start_x + b
                   
                        end_y = k * end_x + b 
                elif direction == 1:
                    points.sort(key=lambda s:s[1]) # y
                    start_y = points[0][1]
                    # print(" k is ", k)
                    # print(" np.isnan(k)", np.isinf(k))
                    if np.isinf(k):
                        start_x = points[0][0]
                    else:
                        start_x = k * start_y + b 
                    end_y = points[-1][1]
                    if np.isinf(k):
                        end_x = points[-1][0]
                    else:
                        end_x = k * end_y + b 
                result_lines.append((start_x, start_y, end_x, end_y, id_num))


        # for sign, metaline in zip(remove, self.metalines):
        #     if not sign:
        #         id_num, direction,_,_,start_x, start_y, end_x, end_y = metaline

        #         result_lines.append((start_x, start_y, end_x, end_y, id_num))
        print("length of result line = ", len(result_lines))
        return result_lines

        # test
        # result_img = np.zeros((origin_img.shape),dtype=np.uint8)
        # for chain in edge_chain:
        #     for col,row in chain:
        #         result_img[row, col] = 255
        # print("done")
        # print('length edge_chain = {}'.format(len(edge_chain)))
        # cv2.imwrite("./img/testedgemap.jpg", result_img)


