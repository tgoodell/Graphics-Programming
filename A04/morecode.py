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

img1=cv2.imread("scenic3.jpg",0)
img2=cv2.imread("scenic4.jpg",0)

orb=cv2.ORB_create()

kp1=orb.detect(img1)
kp2=orb.detect(img2)

kp1,des1=orb.compute(img1,kp1)
kp2,des2=orb.compute(img2,kp2)

bf=cv2.BFMatcher()

matches=bf.knnMatch(des1,des2,k=2)

# KNN: k-nearest neighbors
# bf: bruteforce

good=[]
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)

img3=cv2.drawMatches(img1,kp1,img2,kp2,good,None)
# cv2.imwrite("matches.png",img3)
show(img3)

points1=[]
points2=[]

for m in good:
    pt1=kp1[m.queryIdx].pt
    pt2 = kp2[m.queryIdx].pt
    points1.append(pt1)
    points2.append(pt2)

H,_=cv2.findHomography(np.float32(points2),np.float32(points1),np.float32(points2),np.float32(points1),cv2.RANSAC,5.0)

# Get Gaussian Kernel
# cv2.getGaussianKernel(size,sigma)
# sigma=-1 for default
# k@k.T multiply 1x9 by 9x1, transpose