import numpy as np
import cv2

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

def getPascalRow(x):
    if x==1:
        return [1]
    r=getPascalRow(x-1)
    left=[0]+r
    right=r+[0]
    return [i+j for i,j in zip(left,right)]

img=cv2.imread("image1.png")

# Darken:  kernel=np.float64([[0.9]])
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
kernel=kernel/kernel.sum()
out=cv2.filter2D(img,-1,kernel)
show(out)

m=np.float64([getPascalRow(75)])
kernel=m.T@m
kernel/=kernel.sum()
out=cv2.filter2D(img,-1,kernel)
show(out)
kernel*=255
out=cv2.resize(kernel,(500,500),interpolation=cv2.INTER_LANCZOS4)
show(out)

kernel=np.float64([[0,0,0],[1,0,-1],[0,0,0]])
out=np.abs(out)
Ix=cv2.filter2D(img,cv2.CV_64F,kernel.T)
Iy=cv2.filter2D(img,cv2.CV_64F,kernel)
I=np.sqrt(Ix*Ix+Iy*Iy)
show(np.abs(I))
cv2.imwrite("output.png",I)
