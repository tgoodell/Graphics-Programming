import numpy as np
import cv2

img=cv2.imread("input/input.jpg")

def normalize(img):
    img=img-np.min(img) #the smallest value is 0
    img=img/np.max(img) #the largest value is 1
    img*=255.99
    return np.uint8(img)

if np.min(img)<0 or np.max(img)>255:
    img=normalize(img)
    print("Had to normalize image")

def greyscale(img):
    grey = img[:, :, 1]
    grey[grey < 80] = 0
    grey[grey > 80] = 255
    return grey

def desat(img):
    v=img[:,:,1]
    grey=v[:,:,None]
    print(grey.shape)

img=cv2.imread("input/input-selfie.jpg")
desat(img)
img=greyscale(img)

cv2.imwrite("ouput.jpg",img)