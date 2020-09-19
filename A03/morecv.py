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

print(dir())
img=cv2.imread("image1.png")
kernel=np.float64([[-1/8,-1/8,-1/8],
                    [-1/8,2,-1/8],
                   [-1/8,-1/8,-1/8]])

out=normalize(cv2.filter2D(img,cv2.CV_64F,kernel))
show(img)
show(out)
out=cv2.filter2D(img,-1,kernel)
show(img)
show(out)