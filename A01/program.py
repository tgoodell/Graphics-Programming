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
    grey[grey < 100] = 0
    grey[grey > 100] = 255
    return grey

# def contrast(img):
    # (V-128)*.5+128
    # contra = img[:, :, 0]
    # contra=(contra-128)*.5+128
    # return contra

def contrast(img, factor):
    contra = img[:, :, :]
    contra=1.0*(contra-128)*factor+128
    return contra

def tint(img, color, percent):
    tint=127

    # initial
    g=img[:, :, 1]
    b=img[:, :, 0]
    r=img[:, :, 2]

    if color=="blue":
        g = (1 - percent) * b + percent * tint

    if color=="green":
        g = (1 - percent) * g + percent * tint

    if color=="red":
        r = (1 - percent) * r + percent * tint

    img[:, :, 1] = b
    img[:, :, 0] = g
    img[:, :, 2] = r
    return img

def desaturate(img):
    v=img[:,:,1]
    grey=v[:,:,None]
    print(grey.shape)

img=cv2.imread("input/input-selfie.jpg")
img=tint(img, "green", .7)

cv2.imwrite("ouput.jpg",img)