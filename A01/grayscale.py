import numpy as np
import cv2

def show(img,wait=0,destroy=True):
    if img.min()<0 or img.max()>255:
        img=normalize(img)
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

def normalize(img):
    out=img*1.0
    out-=out.min()
    out/=out.max()
    out*=255.9999
    return np.uint8(out)

img=cv2.imread("PXL_20201202_212501647.jpg");
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
grayscaleImg=0.2126*r+0.7152*g+0.0722*b
show(grayscaleImg)