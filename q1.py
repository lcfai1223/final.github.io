import quopri
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from math import sqrt

img_prefix = './Q1_Image/'
x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

def get_gaussian_blur_img():
    img = cv2.imread(img_prefix + 'edge.png', 0)
    # For debug
    # img = img[0:20, 0:20] 
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # For getting the new matrix with padding width=1
    padding_img = np.zeros((img.shape[0]+2, img.shape[1]+2))

    # The new image should be set as the dtype uint8
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        padding_img[1+i][1:1+img.shape[1]] = img[i]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r1 = np.dot(gaussian_kernel[0], padding_img[i][j:j+3])
            r2 = np.dot(gaussian_kernel[1], padding_img[i+1][j:j+3])
            r3 = np.dot(gaussian_kernel[2], padding_img[i+2][j:j+3])
            new_img[i][j] = r1 + r2 + r3

    new_img = np.asarray(new_img)
    return new_img


def getPxImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./Q1_image/edge.png", img)

    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    qImg = qImg.scaled(600, 600, QtCore.Qt.KeepAspectRatio)
    return QPixmap.fromImage(qImg)

def gaussian_blur(label):
    print("Wait for the Gaussian convolution ...")
    img = get_gaussian_blur_img()
    qImg = getPxImg(img)
    label.setPixmap(qImg)

def get_sobel_img(kernel):
    img = get_gaussian_blur_img()

    # For debug
    # img = img[0:10, 0:10] 
    # For getting the new matrix with padding width=1
    padding_img = np.zeros((img.shape[0]+2, img.shape[1]+2))

    # The new image should be set as the dtype uint8
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        padding_img[1+i][1:1+img.shape[1]] = img[i]

    for i in range(img.shape[0]):
        tmp_row = []
        for j in range(img.shape[1]):
            r1 = np.dot(kernel[0], padding_img[i][j:j+3])
            r2 = np.dot(kernel[1], padding_img[i+1][j:j+3])
            r3 = np.dot(kernel[2], padding_img[i+2][j:j+3])
            if((r1 + r2 + r3) < 0):
                new_img[i][j] = 0
            else:
                new_img[i][j] = r1 + r2 + r3

    new_img = np.asarray(new_img)
    return new_img

def sobel_x(label):
    print("Wait for the sobel_x convolution ...")
    img = get_sobel_img(x_kernel)
    qImg = getPxImg(img)
    label.setPixmap(qImg)

def sobel_y(label):
    print("Wait for the sobel_y convolution ...")
    img = get_sobel_img(y_kernel)
    qImg = getPxImg(img)
    label.setPixmap(qImg)

def magnitude(label):
    print("[0/2] Wait for the sobel_x convolution ...")
    sobel_img_x = get_sobel_img(x_kernel)
    print("      Finish calculating ... \n")

    print("[1/2] Wait for the sobel_y convolution ...")
    sobel_img_y = get_sobel_img(y_kernel)
    print("      Finish calculating ... \n")

    width = sobel_img_x.shape[0]
    height = sobel_img_x.shape[1]
    print("[2/2] Wait for magnitude calculating ...")

    new_img = np.zeros((width, height), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            new_img[i][j] = sqrt(sobel_img_x[i][j]**2 + sobel_img_y[i][j]**2)
    
    qImg = getPxImg(new_img)
    label.setPixmap(qImg)

    # cv2.imshow("image", new_img)
    # print("      Finish calculating ... \n")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()