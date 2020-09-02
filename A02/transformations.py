import numpy as np
import cv2

img=cv2.imread("image2.png",0)

out=img*0

h,w=img.shape[:2]

#for j in range(h):
#    for i in range(w):
#        if 0<=i-10<h:
#            out[j,i]=img[j,i-10]

print(img[[3,5,4,3],[7,5,6,8]])

Y,X=np.mgrid[:h,:w]
Y=Y.ravel() # unravel 2D array into 1D array.
X=X.ravel()
print(Y)

angle=.3
Xp=X*np.cos(angle)-Y*np.sin(angle)
Yp=X*np.sin(angle)+Y*np.cos(angle)
goodTransformationMap=(0<=Xp) * (Xp<w) * (0<=Yp) * (Yp<h)
Xp=np.uint16(Xp)
Yp=np.uint16(Yp)
out[Y[goodTransformationMap],X[goodTransformationMap]]=img[Yp[goodTransformationMap],Xp[goodTransformationMap]]

cv2.imwrite("output.png",out)