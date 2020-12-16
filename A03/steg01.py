import numpy as np
import cv2

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

def normalize(img):
    out=img*1.0
    out-=out.min()
    out/=out.max()
    out*=255.999
    return np.uint8(out)

# print(dir(cv2))
img1=cv2.imread("steg01.png")
img2=cv2.imread("original.jpg")

red=img1[:,:,2]
green=img1[:,:,1]
blue=img1[:,:,0]

out=img1*0;
out=green-red

show(out)