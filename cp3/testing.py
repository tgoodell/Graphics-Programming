import cv2
import numpy as np
import math

DEBUG=True

def show(img,title="image",wait=True):
    d=max(img.shape[:2])
    if d>1000:
        step=int(math.ceil(d/1000))
        img=img[::step,::step]
    if not DEBUG:
        return
    if np.all(0<=img) and np.all(img<256):
        cv2.imshow(title,np.uint8(img))
    else:
        cv2.imshow(title,normalize(img))
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)

def normalize(img):
    img_copy=img*1.0
    img_copy-=np.min(img_copy)
    img_copy/=np.max(img_copy)
    img_copy*=255.9999
    return np.uint8(img_copy)

img=cv2.imread("input_erosion_safe.png",0)

# img[400:800,1000:1400]=0
patch=img[400:800,1000:1400]

patch[:,:]=int(np.average(patch))

show(img)

# find standard deviation of each patch to find optimal places or use average