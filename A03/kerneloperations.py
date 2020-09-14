import numpy as np
import cv2, math

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

img=cv2.imread("image1.png")
h,w=img.shape[:2]
out=img[:-2,:-2,:]*0

kernel=np.ones((3,3),dtype=np.float64)/9
imgSwatch=img[:3,:3]
print((imgSwatch*kernel).sum())

#for i in range(w-2):
 #   for j in range(h-2):
  #      img[j][i]
   #     imgSwatch = img[j:j+3, i:i+3]
    #    out[j][i]=(imgSwatch*kernel).sum()

imgStack=[]
for i in range(3):
    for j in range(3):
        imgStack.append(img[j:j+h-2,i:i+w-2])

block=np.dstack(imgStack)
out=np.uint8((kernel.ravel()).sum(axis=0))


show(out)