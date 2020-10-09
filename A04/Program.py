import numpy as np
import cv2
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

img1=cv2.imread("input/lake1.jpg")
img2=cv2.imread("input/lake2.jpg")
#img1=cv2.resize(img1,(0,0),fx=.4,fy=.4)
#img2=cv2.resize(img2,(0,0),fx=.4,fy=.4)

orb = cv2.ORB_create()

kp1 = orb.detect(img1)
kp2 = orb.detect(img2)

kp1,des1 = orb.compute(img1,kp1)
kp2,des2 = orb.compute(img2,kp2)
# ~ for d in des1:
    # ~ print(d)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2,k=2)

#BF: BRUTE FORCE
#KNN: k-Nearest Neighbors


# ~ good=[m[0] for m in matches]
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        print(des1[m.queryIdx])
        print(des2[m.trainIdx])
        print()
        good.append(m)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)
cv2.imwrite("matches.png",img3)
show(img3)

points1=[]
points2=[]
for m in good:
    pt1=kp1[m.queryIdx].pt
    pt2=kp2[m.trainIdx].pt
    points1.append(pt1)
    points2.append(pt2)

H,_=cv2.findHomography(np.float32(points2), np.float32(points1), cv2.RANSAC, 5.0)
T=np.array([[1,0,0],[0,1,100],[0,0,1]])

img1=cv2.imread("input/lake1.jpg")
img2=cv2.imread("input/lake2.jpg")
h,w=img1.shape[:2]

target_size=(w,h)
out1=cv2.warpPerspective(img1,T,dsize=target_size)
out2=cv2.warpPerspective(img2,T@H)
mask2=cv2.warpPerspective(img2*0+255,T@H)
show(mask2)

mask2=cv2.erode(mask2,np.ones((101,101),np.uint8))
cv2.GaussianBlur(mask2,(101,101),-1)/255.0
show(mask2)
#out1[where out1 isnt]=out2[where out1 isnt]

comp=out2*mask2*(1-mask2)
show(comp)
cv2.imwrite("pano.png",comp)


diff=out1*1.0-out2
diff**=2

cost=diff*1.0
kernel=np.ones(3,np.float64)
for i in range(1,len(cost)):
    minAbove=cv2.errode(cost[i-1],kernel).T[0]
    cost[i]+=minAbove[:,0]

show(cost)
y=len(cost)-1
x=np.argmin(cost[y])
seam=[(x,y)]
while len(seam)<len(cost):
    x,y=seam[-1]
    newY=y-1
    newX=x+np.argmin(cost[newY,x-1:x+2])-1
    seam.append((newX,newY))

diff=normalize(diff)
for x,y in seam:
    diff[x,y]=255
show(diff)

# diff=normalize(img2*1.0-cv2.warpPerspective(img1,HI,(0,0)))
# show(diff)
# ~ h,w,=img1.shape[:2]
# ~ points=np.float64([[[0,0],[w-1,0],[0,h-1],[w-1,h-1]]])
# ~ points=np.hstack((points,cv2.perspectiveTransform(points, H)))

# ~ minx=np.min(points[0,:,0])
# ~ maxx=np.max(points[0,:,0])
# ~ miny=np.min(points[0,:,1])
# ~ maxy=np.max(points[0,:,1])
# ~ target_size=(int(maxx-minx),int(maxy-miny))
# ~ translate=np.float64([[1,0,-minx],[0,1,-miny],[0,0,1]])
# ~ out1=cv2.warpPerspective(img2,translate.dot(H),target_size)
# ~ mask=cv2.warpPerspective(np.uint8(img2*0+255),translate.dot(H),target_size)
# ~ out2=cv2.warpPerspective(img1,translate,target_size)
# ~ diff=normalize((out1-1.0*out2)**2)
# ~ cv2.imwrite("diff.png",diff)
# ~ cv2.imwrite("out1.png",out1)
# ~ cv2.imwrite("out2.png",out2)

# ~ #erode mask
# ~ show(mask)
# ~ transition_zone=129
# ~ kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(transition_zone,transition_zone))
# ~ mask = cv2.erode(mask,kernel,iterations = 1)
# ~ show(mask)
# ~ #blur mask
# ~ mask=cv2.blur(mask,(transition_zone,transition_zone))


# ~ show(mask)
# ~ out=mask/255*out1+(1-mask/255)*out2


# ~ show(out)
# ~ cv2.imwrite("out.png",out)

