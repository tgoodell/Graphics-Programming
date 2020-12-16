import numpy as np
import cv2
import time
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
        print("normalized version")
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

r=86
img=cv2.imread('circles.png')
h,w=img.shape
Y,X=np.mgrid[:h,:w]

votes=np.uint32(img*0)
kernel=np.float64([[0,0,0,],[1,0,-1],[0,0,0]])
Ix=cv2.filter2D(img,cv2.CV_64F,kernel.T)
Iy=cv2.filter2D(img,cv2.CV_64F,kernel)
I=np.sqrt(Ix*Ix+Iy*Iy)
I/=I.max()
y,x=np.where(I>.1)
for i,j in zip(x,y):
    f=I(np.hypot(X-i,Y-j))
    f[f<0]=0
    votes+=f

    cv2.circle(votes,(i,j),r,1)
print(x,y)