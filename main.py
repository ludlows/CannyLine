
import cv2
import numpy as np


def main():
    img_path = "C:\\Users\\Administrator\\Downloads\\lena30.jpg"
    img = cv2.imread(img_path, 0)
    print(img.shape)


if __name__  == "__main__":
    main()