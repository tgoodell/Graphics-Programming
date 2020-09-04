import numpy as np
import cv2

img=cv2.imread("input/image1.png")

def normalize(img):
    img=img-np.min(img) #the smallest value is 0
    img=img/np.max(img) #the largest value is 1
    img*=255.99
    return np.uint8(img)

if np.min(img)<0 or np.max(img)>255:
    img=normalize(img)
    print("Had to normalize image")

def greyscale(img):
    b = img[:, :, 0]*0.1
    g = img[:, :, 1]*0.7
    r = img[:, :, 2]*0.2
    img=b+g+r
    return img

def blackWhite(img, threshold):
    bw = img[:, :, 1]
    bw[bw < threshold] = 0
    bw[bw > threshold] = 255
    return bw

def desaturate(img ,percent):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    b = b * percent
    g = g * percent
    r = r * percent

    img=b+g+r
    return img

def contrast(img, factor):
    contra = img[:, :, :]
    contra=1.0*(contra-128)*factor+128
    return contra

def tint(img, color, percent):
    tint=127
    g=img[:, :, 1]
    b=img[:, :, 0]
    r=img[:, :, 2]

    if color=="blue":
        g = (1 - percent) * b + percent * tint

    if color=="green":
        g = (1 - percent) * g + percent * tint

    if color=="red":
        r = (1 - percent) * r + percent * tint

    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

pic_2_1=greyscale(img)

cv2.imwrite("output/pic_2_1.png",pic_2_1)