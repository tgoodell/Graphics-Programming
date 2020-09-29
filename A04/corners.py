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

img=cv2.imread("house.jpeg",0);
# img=cv2.resize(img,(0,0),fx=.5,fy=.5)

'''
# Cool Edges
kernel=np.array([[1,2,1],
                [0,0,0],
                [-1,-2,-1]])
out=cv2.filter2D(img,cv2.CV_64F,kernel)
blurredImg=cv2.GaussianBlur(img,(7,7),sigmaX=-1)
Iy=cv2.filter2D(blurredImg,cv2.CV_64F,kernel)
Ix=cv2.filter2D(blurredImg,cv2.CV_64F,kernel.T)

I=(Ix*Ix+Iy*Iy)**.5

# Another

edges=cv2.Canny(img,100,200)
'''

kernel=np.array([[1,2,1],
                [0,0,0],
                [-1,-2,-1]])
out=cv2.filter2D(img,cv2.CV_64F,kernel)
blurredImg=cv2.GaussianBlur(img,(7,7),sigmaX=-1)
Iy=cv2.filter2D(blurredImg,cv2.CV_64F,kernel)
Ix=cv2.filter2D(blurredImg,cv2.CV_64F,kernel.T)

I=(Ix*Ix+Iy*Iy)**.5
angle=np.arctan2(Iy,Ix)
edges=cv2.Canny(img,100,200)



show(edges)
show(angle)
show(Ix)
show(Iy)
show(I)
show(Ix*Iy)

dst=cv2.cornerHarris(blurredImg,2,3,0.04)
out=img*0
out[dst>dst.max()/10]=255
show(out)

colorImg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
mout=cv2.goodFeaturesToTrack(blurredImg,300,0.1,25)
for point in mout:
    x,y=point[0]
    print(x,y)
    cv2.circle(colorImg,(x,y),3,(0,255,0),-1)

show(colorImg)