
import cv2
import os
import numpy as np

from cannypf import CannyPF, comp_edge_chain, color_imwrite
from cannyline import MetaLine

def main():
    img_path = r"./img/test07.jpg"
    img = cv2.imread(img_path, 0)
    # print('image shape', img.shape)
    # # compute edge map
    # cannypf = CannyPF(3, 70, img)
    # edgemap = cannypf.comp_edge_map()
    # # cv2.imwrite("C:\\Users\\Administrator\\Downloads\\lena30py.jpg", edgemap)
    # # line chainner , remove noise line
    # edge_chain = comp_edge_chain(img, edgemap)
    # print("computed edge chain")
    # # print(edge_chain)
    # # show image
    # result_img = np.zeros((img.shape),dtype=np.uint8)
    # for chain in edge_chain:
    #     for col,row in chain:
    #         result_img[row, col] = 255
    # cv2.imwrite("C:\\Users\\Administrator\\Downloads\\CannyPFpy.jpg", result_img)
    mtline = MetaLine()
    lines = mtline.mtline_detect(img, 8,1)

    out = 255* np.ones(img.shape, dtype=np.uint8)
    for start_x, start_y, end_x, end_y, _ in lines:
        cv2.line(out, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,0,0),thickness=1, lineType=cv2.LINE_AA)
    name = img_path.split(".")[:-1]
    name = ".".join(name)
    name += "-out.jpg"
    cv2.imwrite(name, out)
    # mtline.getInfo(img,1,1,0.125)
    # cv2.imwrite(r"./img/cannypf.jpg", mtline.canny_edge)

def main2():
    img_path = r"./img/test07.jpg"
    img = cv2.imread(img_path, 0)
    cannypf = CannyPF(3, 70, img)
    edgemap = cannypf.comp_edge_map()
    edge_chain = comp_edge_chain(img, edgemap)
    shape = list(img.shape) + [3]
    name = img_path.split(".")[:-1]
    name = ".".join(name)
    name += "-color-out.jpg"
    color_imwrite(edge_chain, shape, name)

def statistic(dir_name, prefix):
    filenames = os.listdir(dir_name)
    mtline = MetaLine()
    line_length = []
    line_num = []
    for filename in filenames:
        print(filename)
        img = cv2.imread(os.path.join(dir_name, filename), 0)
        out = 255* np.ones(img.shape, dtype=np.uint8)
        lines = mtline.mtline_detect(img, 2, 1)
        length = 0
        for start_x, start_y, end_x, end_y, _ in lines:
            cv2.line(out, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,0,0),thickness=1, lineType=cv2.LINE_AA)
            length += np.sqrt((start_x - end_x)**2 + (start_y-end_y)**2)
        line_num.append(len(lines))
        line_length.append(length)
        
        print("total length is = ", length)
        print("num is = ", len(lines))
        print('avg length', length / len(lines))
        name = filename.split(".")[:-1]
        name = ".".join(name)
        name += "-out.jpg"
        name = prefix + name
        name =  os.path.join('./out/', name)
        print(name)
        cv2.imwrite(name, out)

def demo():
    img_path = r"./img/demo.jpg"
    img = cv2.imread(img_path, 0)
    mtline = MetaLine()
    lines = mtline.mtline_detect(img, 8,1)

    out = 255* np.ones(img.shape, dtype=np.uint8)
    for start_x, start_y, end_x, end_y, _ in lines:
        cv2.line(out, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,0,0),thickness=1, lineType=cv2.LINE_AA)
    name = img_path.split('/')[-1]
    name = name.split(".")[-2]
    name += "-out.jpg"
    print(name)
    name =  os.path.join('./out/', name)
    print(name)
    print(cv2.imwrite(name, out))
    



if __name__  == "__main__":
    demo()