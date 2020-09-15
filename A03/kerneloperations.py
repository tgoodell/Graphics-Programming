import numpy as np
import cv2

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

img=cv2.imread("image1.png",0)
h,w=img.shape[:2]
out=img[:-2,:-2]*0
kernel=np.ones((3,3),dtype=np.float64)/9
show(img)
for i in range(w-2):
    for j in range(h-2):
        imgSwatch=img[j:j+3,i:i+3]
        out[j][i]=(imgSwatch*kernel).sum()
show(out)
imgStack=[]
for i in range(3):
    for j in range(3):
        region=img[j:h-2+j,i:w-2+i]
        imgStack.append(region)
block=np.dstack(imgStack)
out=np.uint8((block*kernel.ravel()).sum(axis=2))
show(out)
show(block.max(axis=2))
show(block.min(axis=2))
show(np.median(block,axis=2))