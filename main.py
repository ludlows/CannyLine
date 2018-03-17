
import cv2
import numpy as np

from cannypf import CannyPF

def main():
    img_path = "C:\\Users\\Administrator\\Downloads\\lena30.jpg"
    img = cv2.imread(img_path)
    print('image shape', img.shape)
    # compute edge map
    cannypf = CannyPF(3, 70, img)
    edgemap = cannypf.comp_edge_map()
    cv2.imwrite("C:\\Users\\Administrator\\Downloads\\lena30py.jpg", edgemap)
    # line chainner , remove noise line
    


if __name__  == "__main__":
    main()